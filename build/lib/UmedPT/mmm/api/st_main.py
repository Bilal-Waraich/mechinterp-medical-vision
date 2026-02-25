import json
from pathlib import Path

import numpy as np

from mmm.logging.st_ext import (
    download_zip_to_workdir,
    m3_image_from_disk,
    st,
    st_batchselector,
    st_download_sidebar,
    st_groupselector,
)
from mmm.settings import mtl_settings


def view_single_prediction(logzip: str, local_workdir: str = "/ephemeral/"):
    download_zip_to_workdir(logzip, workdir := Path(local_workdir))
    with st.expander(label="Local working directory"):
        st.success(f"{workdir.absolute()}")
    selected_group = st_groupselector([p for p in workdir.iterdir() if p.is_dir()])

    if (batch_json_path := workdir.joinpath("batch_info.json")).exists() and "group_id_to_index" in (
        batch_info := json.loads(batch_json_path.read_text())
    ):
        group_name = selected_group.name
        group_index = batch_info["group_id_to_index"][group_name]
        graphs = {key: np.array(batch_info["graphs"][key][group_index]) for key in batch_info["graphs"]}
    else:
        graphs = None

    m3_image_from_disk(st_batchselector(selected_group), graphs=graphs)


def predictions_viewer():
    workdir = st_download_sidebar()
    if (workdir is not None) and (group_folders := [p for p in workdir.iterdir() if p.is_dir()]):
        with st.sidebar:
            selected_group = st_groupselector(group_folders)

        m3_image_from_disk(st_batchselector(selected_group))


def readme():
    st.markdown("run `m3 --help` in a terminal for command line usage instructions.")
    st.markdown(Path(__file__).parent.parent.parent.joinpath("README.md").read_text())


if __name__ == "__main__":
    st.set_page_config(page_title="M3 Streamlit App", layout="wide", page_icon="🏥")

    if "logzip" in st.query_params:
        view_single_prediction(st.query_params["logzip"])
    else:
        pages = {
            "0_Readme": readme,
            "1_Predictions Viewer": predictions_viewer,
        }

        with st.sidebar:
            if mtl_settings.kv.ping():
                st.success("Connected to DB")
            else:
                st.error("Not connected to DB")
            page_names = sorted(list(pages.keys()))
            page = st.selectbox("Select a page", options=page_names, index=1 if "logzip" in st.query_params else 0)

        pages[page]()
