# PTACO
This is an implementation of the Parallel Tempering-based ACO (PTACO) as
described in 
_Skinderowicz, Rafał. "Improving ACO Convergence with Parallel
Tempering.".

TODO


# Why Python and not C?

Generally, because Python code is easier to write and read.
The main drawback is, obviously, its poor performance, especially for
scientific applications which are typically computation-intensive.
Fortunately, thanks to a great PyPy implementation of Python it is possible to
speed up the execution considerably with little to no source code
modifications.

The resulting execution speed is still probably about 3~4 times slower than C version
but that this factor is much easier to accept than 30x or 40x.

# Running

It can be easily run using Python:

./main.py --problem=tsp/kroA100.tsp   --alg=ptaco  --trials=1 --iter_rel=100 --out_dir=results/ --pheromone=std --iter_between_exchange=100

However, I recommend running it with PyPy which uses JIT to speed up the
execution by a factor of 10.

You don't have to install PyPy, a portable distribution is fine
https://github.com/squeaky-pl/portable-pypy#portable-pypy-distribution-for-linux


~/pypy-5.7.1-linux_x86_64-portable/bin/pypy --jit trace_limit=1000 --jit decay=1000 ./main.py --problem=tsp/kroA100.tsp   --alg=ptaco  --trials=1 --iter_rel=1000 --out_dir=results/ --pheromone=std --iter_between_exchange=100


This will print some summary to the screen and write more detailed results to
"./results" directory (it should be created prior to running the program).
The detailed results are written in JSON format.


# License

The source code is licensed under the [MIT
License](http://opensource.org/licenses/MIT):

Copyright 2017 Rafał Skinderowicz (rafal.skinderowicz@us.edu.pl)

Permission is hereby granted, free of charge, to any person obtaining a copy of
this software and associated documentation files (the "Software"), to deal in
the Software without restriction, including without limitation the rights to
use, copy, modify, merge, publish, distribute, sublicense, and/or sell copies
of the Software, and to permit persons to whom the Software is furnished to do
so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING
FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER
DEALINGS IN THE SOFTWARE.
