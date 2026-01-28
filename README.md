# Launching details
## Setting up the project
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

Actually if we would like to get the different export file format then we can pass it in launching command:

`python supports_6.py <here goes the file format>` 

example: `python supports_6.py stl`

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
