## Nodes

### Backbone features 

Node 3512:
- fires on all datasets 
- The feature seems to only focus on the shoulders and focuses mor on  the letters as almost each one has some sort of L or R
- It is confirmed that it gets activated mostly by the corners. 
- conclusion: it is not a universal feature. 
- AUROC is ~0.56

Node 2640:
- fires on all datasets
- Looks at the left side and of the ribs but still very furthest patches on the image
- I do not think that they will be any meaningful reports 
- AUROC is ~0.56

Node 3321:
- similar to node 2640 
- focuses on bottom left of the screen. 
- also focuses on the corner of the images 

Node 1969:
- same results

Node: 1063:
- almost identical to 2640

Node 3832:
- almost identical to 3512


## Healthy Lungs

Node 818:
- focuses on the core spinal area. 
- looks also a bit to the right and to the left but it is predominantly in the middle. 


Node 1384:
- confusing as it seems to be looking for everything but the region of feature 818
- seems to be the anti feature of 818. 
- still seems to focus the lower brighter spots of the image
- this latent cleanly separates COVID-positive from COVID-negative chest X-rays using only its activation magnitude

Node 884:
- focuses on the upper ribs both left and right 

## Pathology-Presence Detectors

Node 3348:
- focuses on the lower 'fleshier part' closer towards the main part of the breast. 
- this could mean that if there some sort of illness that it causes inflamation in the xray ?
- 

Node 2619:
- Similar to node 884 but also 3348. 
- Looks in between and also focuses more on the fleshier part,
- unlike 3348 it focuses only on the bottom right-ish of the lung


## Cross-Illness Structural Patterns

