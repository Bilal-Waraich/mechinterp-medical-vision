from __future__ import annotations

import json
import uuid
from copy import deepcopy
from typing import TYPE_CHECKING, Literal

import aiohttp

from mmm.api.functions.compression import CacheInstances

try:
    from celery.result import AsyncResult
    from fastapi import APIRouter, BackgroundTasks

    from .celery import app
except ImportError:
    if not TYPE_CHECKING:  # makes sure that IDE is not confused
        APIRouter, AsyncResult, app, BackgroundTasks = None, None, None, None


import logfire
from m3_sdk.models import Webhook, WebhookResponse
from m3_sdk.types import CompressType

from mmm.api.api_worker import (
    CacheSubjects,
    cache_instances,
    cache_subjects,
    finetune,
    get_model_info,
    get_task_ids,
    predict,
)
from mmm.api.data import KVReprCohort, ReprDataset
from mmm.api.evaluation import TaskDefinition
from mmm.api.Finetuner import FineTuner
from mmm.api.functions.deeplearning import Finetune, Predict
from mmm.api.models import MSubject
from mmm.api.mtl_adapter import LabelingConfig
from mmm.BaseModel import BaseModel
from mmm.event_selectors import FixedEventSelector
from mmm.settings import mtl_settings


def _get_dataset_key(dataset_name: str):
    return f"datasets:{dataset_name}"


class DLModel:
    class Config(BaseModel):
        category: str = "M3_PEFT"
        pixel_confidence_threshold: float = 0.5

        finetune_config: FineTuner.Config = FineTuner.Config(
            mtl_validation_selector=FixedEventSelector(at_iterations=[])  # never validate
        )

    def __init__(self, prefix: str, settings_kv_key="m3api:settings") -> None:
        """
        The prefix indicates the relative path of the model in the API such as /deeplearning_model.
        """
        self.prefix, self.settings_kv_key = prefix, settings_kv_key
        self.tag = "PEFT"

        self.router = APIRouter()
        self.router.post("/inference", tags=[self.cfg.category])(self.inference)
        self.router.post("/add_subjects", tags=[self.cfg.category])(self.add_subjects)
        self.router.get("/dataset", tags=[self.cfg.category])(self.get_dataset_info)
        self.router.post("/train", tags=[self.cfg.category])(self.train)
        self.router.get("/job_state", tags=[self.cfg.category])(self.get_job_state)
        self.router.get("/info", tags=[self.cfg.category])(self.info)

        # Compressing
        self.router.post("/compress", tags=[self.cfg.category])(self.compress_subjects)
        self.router.get("/compress", tags=[self.cfg.category])(self.get_representations)

        # Functional
        self.router.post("/predict", tags=[f"{self.cfg.category} Functionals"])(self.api_predict)
        self.router.post("/cache_instances", tags=[f"{self.cfg.category} Functionals"])(self.api_cache_instances)

        # Settings
        settings_tag = f"Settings {self.cfg.category}"
        self.router.get("/settingsschema", tags=[settings_tag])(self.get_settings_schema)
        self.router.get("/settings", tags=[settings_tag])(self.get_current_settings)
        self.router.post("/settings", tags=[settings_tag])(self.set_settings)

    @property
    def cfg(self) -> Config:
        if not mtl_settings.kv.exists(self.settings_kv_key):
            logfire.info(
                "No settings found at {settings_key}. Creating default settings.", settings_key=self.settings_kv_key
            )
            mtl_settings.kv.set(self.settings_kv_key, DLModel.Config().model_dump_json())
        return self.Config.model_validate_json(mtl_settings.kv.get(self.settings_kv_key).decode())

    async def get_settings_schema(self) -> dict:
        return self.cfg.model_json_schema()

    async def get_current_settings(self) -> DLModel.Config:
        return self.cfg

    async def set_settings(self, body: dict) -> DLModel.Config:
        mtl_settings.kv.set(self.settings_kv_key, self.Config.model_validate(body).model_dump_json())
        return await self.get_current_settings()

    async def get_job_state(self, task_id: str) -> Literal["PENDING", "SUCCESS", "FAILURE"]:
        # return s.kv.get(f"celery-task-meta-{task_id
        return AsyncResult(task_id, app=app).state

    def get_task_definition(self, labeling_config: LabelingConfig, dataset_key: str):
        task_definition = TaskDefinition(labeling=labeling_config, splits=[])

        num_subjects_in_dataset = mtl_settings.kv.scard(dataset_key)
        if num_subjects_in_dataset < (min_num := task_definition.get_minimum_cohort_size()):
            logfire.info(f"{dataset_key} has {num_subjects_in_dataset} subjects, but needs {min_num} to train.")
            return None
        return task_definition

    async def add_subjects(self, subjects: list[MSubject], add_to_dataset: None | str = None):
        r: CacheSubjects.Results = cache_subjects.delay(CacheSubjects.Args(subjects=subjects)).get()
        if add_to_dataset is not None:
            # Labelstudio overwrites the id, so use the original id which is written by Pydantic into the meta field
            subject_ids = [
                subj.id if subj.meta is None else subj.meta.get("original_subject_id", subj.id) for subj in subjects
            ]
            mtl_settings.kv.sadd(add_to_dataset, *subject_ids)
        return r.subject_keys

    async def compress_subjects(
        self,
        subject_keys: list[str],
        compress_type: CompressType = CompressType.token,
        with_labels: None | list[str] = None,
        subject_groups: int = 1,
        batch_size: int = 1,
        num_workers: int = 0,
    ) -> list[str]:
        subject_batches = [
            subject_keys[subject_batch : subject_batch + subject_groups]
            for subject_batch in range(0, len(subject_keys), subject_groups)
        ]
        add_instances_tasks: AsyncResult = [
            cache_instances.delay(
                CacheInstances.Args(
                    for_type=compress_type,
                    subject_keys=ss,
                    with_labels=with_labels,
                    batch_size=batch_size,
                    num_workers=num_workers,
                )
            )
            for ss in subject_batches
        ]
        return [t.id for t in add_instances_tasks]

    async def api_predict(self, args: Predict.Args) -> Predict.Results:
        return predict.delay(args).get()  # pyright: ignore[reportFunctionMemberAccess]

    async def api_cache_instances(self, args: CacheInstances.Args) -> CacheInstances.Results:
        return cache_instances.delay(args).get()  # pyright: ignore[reportFunctionMemberAccess]

    async def inference(
        self,
        subjects: list[MSubject],
        label_config: LabelingConfig,
        model_id: str = "",
        for_labels: None | dict[str, str] = None,
        compress_type: CompressType | None = None,
        checkpoint: str = "latest",
    ):
        async_results = [
            predict.delay(
                Predict.Args(
                    model_id=model_id,
                    subjects=[subj],
                    t=self.cfg.pixel_confidence_threshold,
                    label_config=label_config,
                    for_labels=for_labels,
                    checkpoint=checkpoint,
                    compress_type=CompressType.rgbimage if compress_type is None else compress_type,
                )
            )
            for subj in subjects
        ]
        return {"results": [r.get().predictions[0].model_dump() for r in async_results]}

    async def forward_events_to_webhook(
        self,
        terminate_with_celery_task_id,
        webhook: Webhook,
        eventqueue: str,
    ):
        # For every event in the Redis queue, send it to the webhook.url
        p = mtl_settings.kv.pubsub()
        p.subscribe(eventqueue)
        while AsyncResult(terminate_with_celery_task_id, app=app).state != "SUCCESS":
            if (next_message := p.get_message(timeout=0.2)) is not None:
                if next_message["type"] == "message":
                    data = WebhookResponse(
                        **dict(return_id=webhook.return_id, event=json.loads(next_message["data"].decode()))
                    )
                    if data.event.event_type in webhook.events:
                        # Send the event to the webhook URL
                        try:
                            async with aiohttp.ClientSession() as session:
                                await session.post(
                                    webhook.url,
                                    json=data.model_dump(exclude_none=True),
                                    headers={"Content-Type": "application/json"},
                                )
                        except Exception as e:
                            print(f"Error sending webhook: {e}")
                            return

    async def train(
        self,
        dataset_name: str,
        labeling_config: LabelingConfig,
        background_tasks: BackgroundTasks,
        mode: Literal["retrain"] | Literal["continue"] = "continue",
        loops: int = 1,
        from_compress_type: CompressType = CompressType.rgbimage,
        finetune_config: FineTuner.Config | None = None,
        wandb_args: dict | None = None,
        webhook: Webhook | None = None,
        model_id: str | None = None,
        batch_size: int = 8,
        num_workers: int = 1,
    ):
        if (task_definition := self.get_task_definition(labeling_config, dataset_name)) is None:
            return {}

        cohorts = []
        for label, cfg in labeling_config.get_parsed().items():
            assert len(cfg["to_name"]) == 1
            label_cohort = KVReprCohort.Config(
                batch_size=(batch_size, batch_size),
                num_workers=num_workers,
                labeling_config=labeling_config,
                train_dataset=dataset_name,
                validation_dataset=dataset_name,
                pretraining=ReprDataset.Config(for_input=cfg["to_name"][0], for_label=label),
                length=1000,
                compress_type=from_compress_type,
            )
            cohorts.append(label_cohort)

        finetune_config = deepcopy(self.cfg.finetune_config) if finetune_config is None else finetune_config
        if mode == "retrain":
            finetune_config.load_prefix = ""

        task: AsyncResult = finetune.delay(
            Finetune.Args(
                redis_cohort_configs=cohorts,
                model_id=dataset_name.split(":")[-1] if model_id is None else model_id,
                cfg=finetune_config,
                num_loops=loops,
                wandb_args=wandb_args,
                eventqueue=(training_event_queue := f"training_{uuid.uuid4()}"),
            )
        )
        if webhook is not None:
            background_tasks.add_task(
                self.forward_events_to_webhook,
                task.id,
                webhook,
                eventqueue=training_event_queue,
            )
        return task.id

    async def info(self):
        return {"task_ids": get_task_ids.delay().get(), "model": get_model_info.delay().get()}

    async def get_representations(
        self,
        pattern: str = f"repr:*:{CompressType.token.value}",
        dataset: str | None = None,
        return_values: Literal["only_number", "all_contexts"] = "all_contexts",
    ) -> dict[str, list[str] | int]:
        """
        - First, finds representations using the pattern.
        - Then, if a dataset is given, filters only subjects within that dataset which consists.
          - The dataset should contain subject ids like "Tom", not keys like "subject:Tom".
        - Depending on return_values, returns either the number of contexts or all contexts.

        For example, if a subject a has a representation for the context "(0,)" and the context "(1,)", the return value
        would be {"a": ["(0,)", "(1,)"]}. All returned values are valid Python programs and can be parsed using eval.
        """
        repr_keys = mtl_settings.kv.keys(pattern)
        if dataset is not None:
            repr_keys = [
                key
                for key in repr_keys
                if key.decode().split(":")[1]
                in list(map(lambda x: x.decode(), mtl_settings.kv.smembers(_get_dataset_key(dataset))))
            ]
        with mtl_settings.kv.pipeline() as pipe:
            for subj_key in repr_keys:
                pipe.hkeys(subj_key)
            all_keys = pipe.execute()
        res = {
            repr_key.decode().split(":")[1]: [key.decode() for key in keys]
            for repr_key, keys in zip(repr_keys, all_keys)
        }
        if return_values == "only_number":
            res = {repr_key: len(keys) for repr_key, keys in res.items()}
            return res
        elif return_values == "all_contexts":
            return res
        else:
            raise ValueError(f"Unknown return value {return_values}")

    async def get_dataset_info(self, dataset_key: str):
        return {
            "subject_keys": mtl_settings.kv.smembers(dataset_key),
            "subject_prefix": mtl_settings.subj_prefix,
        }
