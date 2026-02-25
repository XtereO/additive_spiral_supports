# Launching details
## Setting up the project
0. This project is set up on Python 3.13.7. 

1. Cloning the realization of sorting points spirally in the root of the project (additive_spiral_supports):

`git clone https://github.com/XtereO/sorting_points_spirally.git`

2. Initializing virtual environment for in the root of additive_spiral_supports:

`python -m venv venv`

3. Activating virtual environment:

`.\venv\Scripts\activate`

4. Installing dependencies:

```
pip install -e .
pip install -r req.txt
```

5. Bounding sorting_points_spirally:

```
cd sorting_points_spirally
pip install -e .
```

## Launching the project
Before launching be sure that you have the "cat.stl" file in the root of additive_spiral_supports. This file can be found at: https://disk.yandex.ru/d/aLFb94PN93zrDg.  
After setting up the project we can launch the project (in root of additive_spiral_supports):

`python supports_6.py`
Launching parameters:
| Parameter title  | Values | Description | Default value |
|------------------|--------|-------------|---------------|
| spirally_pattern | 1 or 0 | 1 if it needs to connect supports spirally | 1 |
| min_spacing | float | the min distance between two support points | 0.7 |
| support_thickness | float | the thickness of cylinder supports | 0.2 |
| support_cylinder_sections | int | quality of cylinder supports (the less value, the less memory it takes) | 8 |
| support_tree_sections | int | quality of tree supports | 6 |
| support_joint_subdivisions | int | quality of joints for connecting supports and a part (significantly influence on memory) | 4 |
| export_format | string | the file format of model with supports you want to get | stl |

In general launching looks like this:

`python supports_6.py <spirally_pattern> <min_spacing> <support_thickness> <support_cylinder_sections> <support_tree_sections> <support_joint_subdivisions> <export_format>` 

example: `python supports_6.py 1 2 0.2 4 2 2 stl`

The model with supports will be saved in the root of the directory additive_spiral_supports as "supported_model.stl"

## Code details
If you would like to turn off spiral ordering then in the code at line 890 put False value to the variable `spirally`.

If you would like to turn off vitalization of the spiral algorithm then at line 892 put False value as the second variable of function `sort_points_spirally`. 

# Project results
Using the file "cat.stl" gives the following results:

- Before sorting supports points spirally

![alt text](doc_imgs/cat_scaled_before.png)

- After sorting supports points spirally

![alt text](doc_imgs/cat_scaled_after.png)

- Visualization of 3d model

![alt text](doc_imgs/cat_mesh_partial_right.png)
