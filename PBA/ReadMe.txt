This program contains the source code for the article "Numerical test of the PBA periodogram and Period analysis of FRB 20190520B," and is divided into three parts.

The first folder, "DataSet," is the dataset file required for the program to run.

The second folder, "Simulation experiment," contains the simulation experiment program, which is used to compute the PBA search results for different simulation experiments. The parameters for the simulation experiments can be modified in the "if name == 'main':" module within the program. Each simulation with different parameters runs 100 times, and the program returns the sigma, the number of bursts, and the misjudged periods for each PBA search under the different parameters.

The third file, "Period search for FRB 20190520.py," is the period search for FRB 20190520B. This program relies on the data file for FRB 20190520B (the required data files are all in the "Data Set" folder). The results return the sigma for each period search, the phase of each burst, and the observation phase for each test period (for plotting).

--------------------------------------------------------------------------------------------------------------------------------
This project is licensed under the terms of the MIT license.

MIT License

Copyright (c) [2025] [Numerical test of the PBA periodogram and period analysis of FRB 20190520B]

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.