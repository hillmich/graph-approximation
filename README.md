# Code used for my masters thesis

*Physarum polycephalum* a.k.a *slime mould*

## Project Layout

- `src/` contains the actual implementation of stuff
- `requirements.txt` packages required to make the code work
- `matplotlibrc` defaults for plotting diagrams with Matplotlib

The individual source files have documentation on there own.

## Installation and Dependencies

The code should work with Python 3.6 or above. Some distributions do not include Tkinter (only required for `draw.py`) or venv in
their python packages, so you may have to install them separately with your package manager.

I recommend using *venv* (included since 3.3) to manage the dependencies in an isolated environment.

```
$ cd <project folder>
$ python3 -m venv master-convert
$ source master-convert/bin/activate
$ pip3 install --upgrade pip
(master-convert) $ pip3 install -r requirements.txt
```

### Windows

To run this project on Microsoft Windows I recommend installing [WinPython](https://winpython.github.io/).
You will need to install the following packages (through the *WinPython Control Panel*).
Depending your specific versions of Python and WinPython more additional packages might be  required.
 
 - [numpy+mkl](http://www.lfd.uci.edu/~gohlke/pythonlibs/#numpy)
 - [opencv-python](https://pypi.python.org/pypi/opencv-python)
 
 

## Example

As a basic example the following shows how to go from the MATLAB data to clustered graphs. It assumes the *venv* is set up and activated.
```
(master-convert) $ python src/convert_matlab.py --output-dir results/converted/matlab/ data/matlab/2011_09_14_crop_1/
[...]
(master-convert) $ python src/cluster_ahead.py --output-dir results/clustered/matlab_ahead_s2 --look-ahead-steps 2 results/converted/matlab/2011_09_14_crop_1/
[...]
(master-convert) $ python src/draw.py results/clustered/matlab_ahead_s2/2011_09_14_crop_1/
[...]
```

## Drawing

`draw.py` can navigate through a folder containing multiple graphs using the arrow keys. If the crop you loaded provides cluster information, you can track them by clicking on a node and than navigating.

Pressing *t* while viewing a picture will output TikZ-Code into the `figures`-folder (this does not include background pictures).

## Word Definitions

- *session*: experiment run on one Petri dish, can contain one or more *crops*
- *crop*: data gathered from a single drop placed on Petri dish
- *picture*: a single snapshot capturing the state of the slime mould
- *graph*: network constructed from a *picture*
