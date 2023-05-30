# First breaks picking
This project is devoted to pick waves that are the first to be detected on a seismogram (first breaks, first arrivals).
Traditionally, this procedure is performed manually. When processing field data, the number of picks reaches hundreds of
thousands. Existing analytical methods allow you to automate picking only on high-quality data with a high
signal / noise ratio. 

As a more robust algorithm, it is proposed to use a neural network to pick the first breaks. Since the data on adjacent
seismic traces have similarities in the features of the wave field, **we pick first breaks on 2D seismic gather**, not 
individual traces.

![](https://raw.githubusercontent.com/DaloroAT/first_breaks_picking/main/docs/images/intro_small.PNG)

# Installation

Library is available in PyPI:
```shell
pip install -U first-breaks-picking
```

### Extra data

- To pick first breaks you need 
to [download model](https://oml.daloroserver.com/download/seis/fb.onnx). 

- If you have no seismic data, you can also 
[download small SGY file](https://raw.githubusercontent.com/DaloroAT/first_breaks_picking/main/data/real_gather.sgy).

It's also possible to download them with Python using the following snippet:

```python
from first_breaks.utils.utils import (download_demo_sgy, 
                                      download_model_onnx)

sgy_filename = 'data.sgy'
model_filename = 'model.onnx'

download_demo_sgy(sgy_filename)
download_model_onnx(model_filename)
```

# How to use it

The library can be used in Python, or you can use the desktop application. The former has more flexibility for 
building your own picking scenario and processing multiple files. The difference between the latter is that it 
allows you to work interactively with only one file and has better performance in visualization.


### Create SGY
We provide several ways to create `SGY` object: from file, `bytes` or `numpy` array. 

From file:
```python
from first_breaks.sgy.reader import SGY

sgy_filename = ...  # put path to your file. 
sgy = SGY(sgy_filename)
```

From `bytes`:
```python
from first_breaks.sgy.reader import SGY

sgy_filename = ...  # put path to your file. 

with open(sgy_filename, 'rb') as fin:
    sgy_bytes = fin.read()

sgy = SGY(sgy_bytes)
```

From `numpy` array:
```python
import numpy as np
from first_breaks.sgy.reader import SGY

num_samples = 1000
num_traces = 48
dt_mcs = 1e3

traces = np.random.random((num_samples, num_traces))
sgy = SGY(traces, dt_mcs=dt_mcs)
```

#### Content of SGY 

Created `SGY` allows you to read traces, get observation parameters and view headers (empty if created from `numpy`)

```python
from first_breaks.sgy.reader import SGY

sgy: SGY = ...  # put here previously created SGY

# get all traces or specific traces limited by time
all_traces = sgy.read()
block_of_data = sgy.read_traces_by_ids(ids=[1, 2, 3, 10],
                                       min_sample=100,
                                       max_sample=500)

# number of traces, values are the same
print(sgy.num_traces, sgy.ntr)
# number of time samples, values are the same
print(sgy.num_samples, sgy.ns) 
# = (ns, ntr)
print(sgy.shape)
# time discretization, in mcs, in mcs, in ms
print(sgy.dt, sgy.dt_mcs, sgy.dt_ms)

# dict with headers in the first 3600 bytes of the file
print(sgy.general_headers)
# pandas DataFrame with headers for each trace
print(sgy.traces_headers.head())
```

### Create task for picking

Next, we create a task for picking and pass the picking parameters to it. They have default values, but for the 
best quality, they must be matched to specific data. You can use the desktop application to evaluate the parameters.
A detailed description of the parameters can be found  in the following chapters.

```python
from first_breaks.sgy.reader import SGY
from first_breaks.picking.task import Task

sgy: SGY = ...  # put here previously created SGY
task = Task(sgy, 
            traces_per_gather=24,
            maximum_time=200)
```

### Pick first breaks

In this step, we use the neural network for picking. If you downloaded the model according to the installation section, 
then pass the path to it. Or leave the path to the model empty so that we can download it automatically.

```python
from first_breaks.picking.task import Task
from first_breaks.picking.picker import PickerONNX

task: Task = ...  # put here previously created task
picker = PickerONNX()
task = picker.process_task(task)

# we can see results of picking
print(task.picks_in_samples)
print(task.picks_in_ms)
print(task.confidence)
```






[//]: # (- We can create it from `numpy` array.)

# Picking process

Neural network process file as series of **images**. There is why **the traces should not be random**,
since we are using information about adjacent traces.

To obtain the first breaks we do the following steps:
1) Read all traces in the file.
![All traces](https://raw.githubusercontent.com/DaloroAT/first_breaks_picking/main/docs/images/full.png)
2) Limit time range by `Maximum time`. 
![Limited by time](https://raw.githubusercontent.com/DaloroAT/first_breaks_picking/main/docs/images/tm_100.png)
3) Split the sequence of traces into independent gathers of lengths `Traces per gather` each. 
![Splitted file](https://raw.githubusercontent.com/DaloroAT/first_breaks_picking/main/docs/images/tm_100_tr_24_24_24_24.png)
4) Apply trace modification on the gathers level if necessary (`Gain`, `Clip`, etc). 
5) Calculate first breaks for individual gathers independently.
![Picked shots](https://raw.githubusercontent.com/DaloroAT/first_breaks_picking/main/docs/images/tm_100_tr_24_24_24_24_picks.png)
6) Join the first breaks of individual gathers.
![Picked file](https://raw.githubusercontent.com/DaloroAT/first_breaks_picking/main/docs/images/tm_100_picks.png)

To achieve the best result, you need to modify the picking parameters.

### Traces per gather

`Traces per gather` is the most important parameter for picking. The parameter defines how we split the sequence of 
traces into individual gathers. 

Suppose we need to process a file with 96 traces. Depending on the value of `Traces per gather` parameter, we will 
process it as follows:
- `Traces per gather` = 24. We will process 4 gathers with 24 traces each. 
![4 full shots](https://raw.githubusercontent.com/DaloroAT/first_breaks_picking/main/docs/images/tm_100_tr_24_24_24_24.png)
- `Traces per gather` = 48. We will process 2 gathers with 48 traces each.
![2 full shots](https://raw.githubusercontent.com/DaloroAT/first_breaks_picking/main/docs/images/tm_100_tr_48_48.png)
- `Traces per gather` = 40. We will process 2 gathers with 40 traces each and 1 gather with the remaining 16 traces. 
The last gather will be interpolated from 16 to 40 traces. 
![2 full + 1 interpolated shots](https://raw.githubusercontent.com/DaloroAT/first_breaks_picking/main/docs/images/tm_100_tr_40_40_16.png)

### Maximum time

You can localize the area for finding first breaks. Specify `Maximum time` if you have long records but the first breaks
located at the start of the traces. Keep it `0` if you don't want to limit traces.

### List of traces to inverse

Some receivers may have the wrong polarity, so you can specify which traces should be inversed. Note, that inversion 
will be applied on the gathers level. For example, if you have 96 traces, `Traces per gather` = 48 and
`List of traces to inverse` = (2, 30, 48), then traces (2, 3, 48, 50, 78, 96) will be inversed.

Notes:
- Trace indexing starts at 1.
- Option is not available on desktop app.


## Recommendations
You can receive predictions for any file with any parameters, but to get a better result, you should comply with the 
following guidelines:
- Your file should contain one or more gathers. By a gather, we mean that traces within a single gather can be 
geophysically interpreted. **The traces within the same gather should not be random**, since we are using information 
about adjacent traces.
- All gathers in the file must contain the same number of traces.
- The number of traces in gather must be equal to `Traces per gather` or divisible by it without a remainder. 
For example, if you have CSP gathers and the number of receivers is 48, then you can set the parameter 
value to 48, 24, or 12.
- We don't sort your file (CMP, CRP, CSP, etc), so you should send us files with traces sorted by yourself. 
- You can process a file with independent seismograms obtained from different polygons, under different conditions, etc., 
but the requirements listed above must be met.
