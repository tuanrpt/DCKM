# Deep Cost-sensitive Kernel Machine Model 
This is an implementation of the Deep Cost-sensitive Kernel Machine (DCKM) model described in the **Deep Cost-sensitive Kernel Machine
for Binary Software Vulnerability Detection** paper.

DCKM model is a combination of a number of diverse techniques, including deep learning, kernel methods, and the new cost-sensitive based approach, aiming to detect efficiently potential vulnerabilities in binary software. 

The overall structure of DCKM model consists of 3 primary elements: an embedding layer for vectorizing machine instructions, a Bidirectional Recurrent Neural Network capable of taking into account temporal information from a sequence of machine instructions, and a novel Cost-sensitive Kernel Machine invoked in the random feature space to predict the vulnerability with minimal cost-sensitive loss.

The model is trained on two binary datasets, **NDSS18** and **6 open-source** which is a new real-world binary dataset whose source code was collected from six open-source projects.

## Instructions
### Datasets
**The statistics of two binary datasets**
<table>
  <tr align="center">
    <th></th>
    <th></th>
    <th><b>#Non-vul</b></th>
    <th><b>#Vul</b></th>
    <th><b>#Binaries</b></th>
  </tr>
  <tr align="center">
    <td rowspan="3"><b>NDSS18</b></td>
    <td><b>Windows</b></td>
    <td>8,999</td>
    <td>8,978</td>
    <td>17,977</td>
  </tr>
  <tr align="center">
    <td><b>Ubuntu</b></td>
    <td>6,955</td>
    <td>7,349</td>
    <td>14,304</td>
  </tr>
  <tr align="center">
    <td><b>Whole</b></td>
    <td>15,954</td>
    <td>16,327</td>
    <td>32,281</td>
  </tr>
  <tr align="center">
    <td rowspan="3"><b>6 open-source</b></td>
    <td><b>Windows</b></td>
    <td>26,621</td>
    <td>328</td>
    <td>26,949</td>
  </tr>
  <tr align="center">
    <td><b>Ubuntu</b></td>
    <td>25,660</td>
    <td>290</td>
    <td>25,950</td>
  </tr>
  <tr align="center">
    <td><b>Whole</b></td>
    <td>52,281</td>
    <td>618</td>
    <td>52,899</td>
  </tr>
</table>

**Data format**

Each dataset folder contains two files: binaries-x-y.data (including functions compiled into binaries under two platforms, Windows and
Ubuntu), and their corresponding labels, labels-x-y.data (where x is 32 or 64, y is windows or ubuntu).

**An example of content of binary files**
> **Note:** We use the special charactor '|' to distinguish *opcodes* from *instruction information*. For example, '131|131,196,8' has an opcode '131' and instruction information '131,196,8'. Additionally, the functions are separated by '-----'.   
```
...
-----
85|85
137|137,229
104|104,0,0,0,0
106|106,1
232|232,252,255,255,255
131|131,196,8
255|255,117,8
232|232,252,255,255,255
131|131,196,4
144|144
201|201
195|195
-----
85|85
137|137,229
131|131,236,16
...
```

**An example of content of label files**
> **Note:** label '0' and '1' represents for a non-vulnerable and vulnerable function respectively.

```
0 0 0 0 0 0 1 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0
```
