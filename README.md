<!--
# pmph-e2024-pub
--PMPH course 2024-25 public repo
-->

# Programming Massively Parallel Hardware (PMPH), Block 1 2024

### We are grateful to Nvidia for awarding us a teaching grant (for the PMPH and DPP courses) that consists of two A100 GPUs. These are now accessible on the Hendrix GPU cluster.

## Course Structure

PMPH is structured to have four hours of (physical) lectures
and four hours of (physical) labs per week; potentially we
will have no lectures in the last few weeks of the course, so you
can concentrate on project work (to be announced).

[Course Catalog Web Page](https://kurser.ku.dk/course/ndak14008u/2024-2025)

### Lectures (zoom links will be posted on Absalon):

* Monday    13:00 - 15:00 (aud - NBB 2.0.G.064/070, Jagtvej 155)
* Wednesday 10:00 - 12:00 (aud - NBB 2.0.G.064/070, Jagtvej 155)

### Labs:

* Monday    15:00 - 17:00 (aud - Aud 02 AKB, Universitetsparken 13)
* Wednesday 13:00 - 15:00 (aud - NBB 2.0.G.064/070, Jagtvej 155)

### Flexible Schedule on Wednesday

We have also reserved room (aud - NBB 2.0.G.064/070, Jagtvej 155)
for Wednesday 15:00 -- 17:00, so that we can stay over if necessary.


### Physical Attendence to Lectures and Labs

The current plan is that everybody will have a physical place
at the lecture and lab. Unless we are forced to move to virtual
teaching, the lectures and labs will not be recorded, so please
plan to attend. If there is strong request, we may stream the
lectures, but without providing any guarantees as to the quality
of streaming.

### Evaluation

Throughout the course, you will hand in three weekly assignments,
which will count for 40\% of the final grade, as follows: the first two
weeklies count for 10\% each and the third one is a "double" assignment
and counts 20\%. In the last month (three weeks) of the course, you will
work on a group project (up to four students per group), and you will submit
the report and accompanying code. The group project will be presented
orally at the exam (i.e., a slide-based presentation) together with the
answers to some individual questions, and this will count for 60\% of
your final grade.

For more information on weekly assignments and (re-)submission, please see [this
section](#weekly-assignments).

For the group project no re-submission is possible; the deadline is the Friday
just before the exam week. Extension may be granted for up until Monday the
exam week (this decision made entirely by Cosmin).

The oral examination will be hold in the exam week (Wednesday, Thursday and Friday if necessary).
The final evaluation will take up to 20 minutes per student, but the whole group will be examined at a time (unless you wish otherwise).

**Weekly assignment and group project hand-in is still on Absalon.**

### Teachers

Teacher: **Cosmin Oancea** ([cosmin.oancea@diku.dk](mailto:cosmin.oancea@diku.dk)).

Teaching assistants (TAs):
* **Anders Holst** ([lietzen@di.ku.dk](mailto:lietzen@di.ku.dk), Discord: `sortraev`),
* **Nikolaj Hinnerskov** ([nihi@di.ku.dk](mailto:nihi@di.ku.dk), Discord: `nikolaj2978`).

Cosmin will conduct lectures and lab sessions; Anders will be grading your
weekly assignments and patrolling the online (Discord/Absalon) discussion forums;
finally, Nikolaj will assist Cosmin in the lab sessions, and may supplement
Anders on the discussion forums from time to time.

### Course Tracks and Resources

All lectures and lab sessions will be delivered in English.  The
assignments and projects will be posted in English, and while you can
chose to hand in solutions in either English or Danish, English is
preferred. All course material except for the hardware book is distributed
via this GitHub page (note: assignment handin is still on Absalon!).

* **The hardware track** of the course covers (lecture) topics related to processor, memory and interconnect design, including cache coherency, which are selected from the book [Parallel Computer Organization and Design, by Michel Dubois, Murali Annavaram and Per Stenstrom,  ISBN 978-521-88675-8. Cambridge University Press, 2012](https://www.cambridge.org/dk/academic/subjects/engineering/computer-engineering/parallel-computer-organization-and-design?format=HB&isbn=9780521886758). The book is available at the local bookstore (biocenter). It is not mandatory to buy it---Cosmin thinks that it is possible to understand the material from the lecture slides, which are detailed enough---but also note that lecture notes are not provided for the hardware track, because of copyright issues.

* **The software track** covers (lecture) topics related to parallel-programming models and recipes to recognize and optimize parallelism and locality of reference.  It demonstrates that compiler optimizations are essential to fully utilizing hardware, and that some optimizations can be implemented both in hardware and software, but with different pro and cons.   [The lecture notes are available here](http://hjemmesider.diku.dk/~zgh600/Publications/lecture-notes-pmph.pdf), and additional (facultative) reading material (papers) will be linked with individual lectures; see Course Schedule Section below.

* **The lab track** teaches GPGPU hardware specifics and programming in Futhark, CUDA, and OpenMP. The intent is that the lab track applies in practice some of the parallel programming principles and optimizations techniques discussed in the software tracks. It is also intended to provide help for the weekly assignment, group project, etc.

## Course Schedule

This course schedule is tentative and will be updated as we go along. 
The links will become functional as we get near to the corresponding date.

The lab sessions are aimed at providing help for the weeklies and
group project.  Do not assume you can solve them without attending
the lab sessions.

| Date | Time | Topic | Material |
| --- | --- | --- | --- |
| 02/09 | 13:00-15:00 | [Intro, Hardware Trends and List Homomorphisms (LH - SFT)](slides/L1-Intro-Org-LH.pdf), Chapters 1 and 2 in [Lecture Notes](http://hjemmesider.diku.dk/~zgh600/Publications/lecture-notes-pmph.pdf) | Facultative material: [Sergei Gorlatch, "Systematic Extraction and Implementation of Divide-and-Conquer Parallelism"](facultative-material/List-Hom/GorlatchDivAndConq.pdf);  [Richard S. Bird, "An Introduction to the Theory of Lists"](facultative-material/List-Hom/BirdThofLists.pdf); [Jeremy Gibons, "The third homomorphism theorem"](facultative-material/List-Hom/GibonsThirdTheorem.pdf) |
| 02/09 | 15:00-17:00 | [Gentle Intro to CUDA](slides/Lab1-CudaIntro.pdf) | [helper CUDA code](HelperCode/Lab-1-Cuda); as facultative material you may consult Cuda tutorials, for example [a very simple one is this one](https://developer.nvidia.com/blog/even-easier-introduction-cuda/) and [a more comprehensive one is this one](https://docs.nvidia.com/cuda/cuda-c-programming-guide/index.html)
| 04/09 | 10:00-12:00 | [List Homomorphism (LH) & Parallel Basic Blocks (SFT)](slides/L2-Flatenning.pdf), Chapters 2 and 3 in [Lecture Notes](http://hjemmesider.diku.dk/~zgh600/Publications/lecture-notes-pmph.pdf) | Facultative material: [Various papers related to SCAN and flattening, but which are not very accessible to students](facultative-material/Flattening) |
| 04/09 | 13:00-15:00 | Lab: Futhark programming, First Weekly | [Futhark code related to the LH lecture](HelperCode/Lect-1-LH); as facultative material: [Parallel Programming in Futhark](https://futhark-book.readthedocs.io/en/latest/), sections 1-4, |
| 04/09 | some time   | [**Assignment 1 handout**](weeklies/weekly-1/) | |
| 09/09 | 13:00-15:00 | [Parallel Basic Block & Flattening Nested Parallelism (SFT)](slides/L2-Flatenning.pdf) | chapters 3 and 4 in [Lecture Notes](http://hjemmesider.diku.dk/~zgh600/Publications/lecture-notes-pmph.pdf) |
| 09/09 | 15:00-17:00 | Lab: [Fun Quiz](slides/Lab2_presentation.pdf); | help with weekly |
| 11/09 | 10:00-12:00 | [Flattening Nested Parallelism (SFT)](slides/L2-Flatenning.pdf) [In-Order Pipelines (HWD)](slides/L3-InOrderPipe.pdf)| Chapter 4 in Lecture Notes, Chapter 3 of "Parallel Computer Organization and Design" Book |
| 11/09 | 13:00-15:00 | Lab: [Reduce and Scan in Cuda](slides/Lab2-RedScan.pdf) | discussing second weekly, helping with the first |
| 11/09 | some time   | [**Assignment 2 handout**](weeklies/weekly-2/) | |
| 16/09 | 13:00-15:00 | [In-Order Pipelines (HWD)](slides/L3-InOrderPipe.pdf), [Optimizing ILP, VLIW Architectures (SFT-HWD)](slides/L4-VLIW.pdf) | Chapter 3 of "Parallel Computer Organization and Design" Book |
| 16/09 | 15:00-17:00 | Lab: [GPU hardware: three important design choices.](slides/Lab2-GPU-HWD.pdf) | helping with weeklies |
| 18/09 | 10:00-12:00 | [Dependency Analysis of Imperative Loops](slides/L5-LoopParI.pdf) | Chapter 5 of lecture Notes |
| 18/09 | 13:00-15:00 |  | helping with the first two weekly assignments.
| 18/09 |  | No new weekly assignment this week; the third will be published next week | |
| 23/09 | 13:00-15:00 | [Demonstrating Simple Techniques for Optimizing Locality](slides/L6-locality.pdf) | Chapter 5 and 6 of Lecture Notes |
| 23/09 | 15:00-17:00 | [**Assignment 3+4 handout**](weeklies/weekly-3-4/) | helping with the weekly assignments. |
| 25/09 | 10:00-12:00 | [Optimizing Locality same idea in other words: Nearest Neighbor, and again Matrix Multiplication and Transposition](slides/L5-LoopParI.pdf) | Chapters 5 and 6 of lecture Notes |
| 25/09 | 13:00-15:00 | Lab: discussing the third assignment | helping with the weekly assignments.
| 30/09 | 13:00-15:00 | [Memory Hierarchy, Bus-Based Coherency Protocols (HWD)](slides/L7-MemIntro.pdf) | Chapter 4 and 5 of "Parallel Computer Organization and Design" Book |
| 30/09 | 15:00-17:00 | Lab: [**Presenting Possible Group Projects**](group-projects/) | discussing group projects, helping with weekly assignments |
| 02/10 | 10:00-12:00 | HWD: [Scalable Coherence Protocols](slides/L8-Interconnect.pdf) | Chapters 5 and 6 of "Parallel Computer Organization and Design" Book |
| 02/10 | 13:00-15:00 | Lab: [**Presenting Possible Group Projects**](group-projects/) | helping with weekly assignments, discussing group projects.
| 07/10 | 13:00-15:00 | HWD: [Scalable Coherence Protocols, Scalable Interconect (HWD)](slides/L8-Interconnect.pdf) [Exercises](hwd-exercises/hwd-coherence-in-exercises.pdf)| Chapters 5 and 6 of "Parallel Computer Organization and Design" Book |
| 07/10 | 15:00-17:00 | Lab: helping with weekly assignments and project |  |
| 09/10 | 10:00-12:00 | [Demonstrating by Exercises the Coherency Protocols and Interconnect material](hwd-exercises/hwd-coherence-in-exercises.pdf) | |
| 09/10 | 13:00-15:00 | | helping with weeklies and project
| 14/10 | 13:00-15:00 | Autumn break (no lecture) | |
| 14/10 | 15:00-17:00 | Autumn break (no lab) | |
| 16/10 | 10:00-12:00 | Autumn break (no lecture) | |
| 16/10 | 13:00-15:00 | Autumn break (no lab) |
| 21/10 | 13:00-15:00 | No lecture | Upon request, may show up and help with group project/weeklies |
| 21/10 | 15:00-17:00 | Lab: Helping with group project and weeklies | |
| 23/10 | 10:00-12:00 | [Inspector-Executor Techniques for Locality Optimizations (SFT)](slides/L8-LocOfRef.pdf) | [Various scientific papers](material/Opt-Loc-Ref) |
| 23/10 | 13:00-15:00 | Lab: help with group project, weeklies |
| 28/10 | 13:00-15:00 | Lecture: helping with group project and weeklies | you may read Tomasulo Algorithm (HWD) from Chapter 3 of "Parallel Computer Organization and Design" Book; [also on slides](slides/L9-OoOproc.pdf) |
| 28/10 | 15:00-17:00 | Lab: Helping with group project, weeklies | |
| 30/10 | 10:00-12:00 | Lecture: helping with group project | |
| 30/10 | 13:00-15:00 | Lab: help with group project |
| 06/11 | whole day | Oral exam | one four-person group will be examined in up to 1 hour and 20 minutes, but all of you will take two-to-three full days.|
| 07/11 | whole day | Oral exam | |
| 08/11 | whole day | Oral exam | |

## Weekly assignments

As mentioned, weekly assignments are **mandatory**, must be solved
**individually**, and make up 40% of your final grade. The first two
weekly assignments (W-assignments) are tentatively planned to be published on
Wednesday of the first and second week. The third (last) assignment will
probably be published on Monday of the fourth week; see the course schedule
section below.

### Submission and resubmission of weekly assignments

*All submissions and resubmissions are to be made on Absalon!*

One resubmission attempt is granted for each weekly assignment, and the
resubmission attempt may be used to solve tasks missing in the original
hand-in and/or to improve on the existing hand-in. Please note that the
first upload made on Absalon *after* the original hand-in deadline will be
considered as the resubmission hand-in!

Given the high number of students this year, feedback may take up to 1.5 weeks
to arrive -- feel free to message Anders if you are curious. The resubmission
deadline is either **two weeks** from the original handin deadline *or* **one
week** from the time of receiving feedback; whichever comes *last*.

Extensions may be granted on weekly assignment (re-)submission deadlines --
please ask Anders if for any reason, personal or otherwise, you need an
extension (no need to involve Cosmin unless you wish to complain about Anders'
decision).

### Weekly 1 (due September 12th)

* [Assignment text](weeklies/weekly-1/assignment1.asciidoc)
* [Code handin](weeklies/weekly-1/w1-code-handin.tar.gz)

### Weekly 2 (due September 23rd)

* [Assignment text](weeklies/weekly-2/assignment2.asciidoc)
* [Code handout](weeklies/weekly-2/w2-code-handin.tar.gz)

### Weekly 3+4 (due October 6th) -- this is a bigger assignment counting as two assignments

* [Assignment text](weeklies/weekly-3-4/assignment3-4.asciidoc)
* [Code handout](weeklies/weekly-3-4/w3-code-handin.tar.gz)


## Group projects (due Friday just before the exam week starts)

Several potential choices for group project may be found in folder `group-projects`, namely

* **You are free to propose your own project, for example from the machine learning field, but please discuss it first with Cosmin, to make sure it is a relevant project, i.e., on which you can apply some of the techniques/reasoning that we have studied in PMPH.**
* [Single Pass Scan in Cuda (basic block of parallel programming)](group-projects/single-pass-scan)
* [GPU Implementation of Linear Recurrences](group-projects/linear-rec)
* [Futhark or Cuda implementation for the Rank-K Search Problem](group-projects/rank-search-k)
* [Fast Sorting Algorithm(s) for GPUs](group-projects/sorting-on-gpu)
* [Bfast: a landscape change detection algorithm (Remote Sensing)](group-projects/bfast)
* [Local Volatility Calibration  (Finance)](group-projects/loc-vol-calib)
* [HP Implementation for Fusing Tensor Contractions (Deep Learning)](group-projects/tensor-contraction): read the paper, implement the technique (some initial code is provided), and try to replicate the results of the paper. Or you can also try to implement a matrix multiplication for 16-bit floats that uses the tensor-core support.

[Here you can find the CUB library and a simple program that utilizes CUB to sort](group-projects/cub-code)

## GPU + MultiCore Machines

All students will be provided individual accounts on a multi-core and GPGPU
machine that supports multi-core programming via C++/OpenMP and CUDA
programming.

The available machines are equipped with top-end A100 GPUs & two AMD EPYC 7352
24-Core CPUs (total 96 hardware threads). You should have access to these
machines from September 2.

More specifically, the GPUs are located on the Futhark servers (access to which
is obtained via the Hendrix gateway; connection guide below), which include:
`hendrixfut01fl` or `hendrixfut03fl` servers, each of which is equipped with an
NVIDIA A100 GPU, on which you can run CUDA programs; and `hendrixfut02fl`, which
has an AMD GPU and hence cannot run CUDA programs (but which can run OpenCL
binaries, e.g. compiled using Futhark).


### Basic Futhark server connection guide

This basic connection guide is subject to change and may contain errors or
have missing information. Please ask Anders if you have trouble logging onto the
servers, or Cosmin, if you suspect the problem is an access/permission issue.

For more comprehensive info on the Hendrix cluster and how to connect, please
see [this](https://diku-dk.github.io/wiki/slurm-cluster).

#### Step 0 -- update ssh config (one-time, optional)

Add the Hendrix gateway server to your ssh config by appending below paragraph:

```
Host hendrix
    HostName hendrixgate
    User <KU-id>
    StrictHostKeyChecking no
    CheckHostIP no
    UserKnownHostsFile=/dev/null
```

to your ssh config file, located in either `$HOME/.ssh/config` for Linux/MacOS,
or `C:/Users/<user>/.ssh/config` for Windows (note: a simple text file with no
file extension). Remember to replace `<KU-id>` in line 3 with your personal
KU-id.

#### Step 1 -- connect to KU-VPN

*Each time* you wish to SSH to the Hendrix cluster/gateway, you must first be
properly connected to KU-VPN. If you get a "Name or service not known" error
from `ssh`, then you are probably not connected to the VPN. See [this guide on
connecting to KU-VPN](https://github.com/diku-dk/howto/blob/main/vpn.md).

#### Step 2 -- connect to Hendrix gateway

If you updated your ssh config as per step 0 and are properly connected to
KU-VPN, you may SSH to the Hendrix gateway using:

```bash
$ ssh hendrix
```

If you skipped step 0, you may need to manually supply appropriate flags to
`ssh` on each login.

##### Step 2b -- setup CUDA dev environment (one-time). Cosmin thinks that this step is NOT NEEDED!

The **first time** you log onto the Hendrix gateway, you must setup the CUDA dev
environment. To do so, permanently update your path by appending below snippet
to your `$HOME/.bash_profile` and/or `$HOME/.bashrc` files:

```bash
export CPATH=/usr/local/cuda/include:$CPATH
export LIBRARY_PATH=/usr/local/cuda/lib64:$LIBRARY_PATH
export LD_LIBRARY_PATH=/usr/local/cuda/lib64/$LD_LIBRARY_PATH
export PATH=/usr/local/cuda/bin:$PATH
```

Finally, reload the affected file/s using `source $HOME/.bash_profile`
and/or `source $HOME/.bashrc` (or simply starting a new bash session).

#### Step 3 -- connect to Futhark machine on the Hendrix cluster

From the Hendrix gateway shell, you may log further onto one of the Futhark
machines using e.g.:

```bash
$ ssh hendrixfut03fl
```

Once logged onto a Futhark server, you may need to load CUDA and/or Futhark
modules using:

```bash
$ module load cuda;
$ module load futhark;
```

Or you may add those at the end of `$HOME/.bash_profile` and/or `$HOME/.bashrc` so you do them only once.


<!-- ##### Alternatively: upload slurm jobs for exclusive execution -->
<!--  -->
<!-- If you prefer (or need) limited-time exclusive access to a GPU -- rather than -->
<!-- logging directly onto a Futhark server along with fellow students -- then you -->
<!-- can submit Slurm jobs to a job queue from the Hendrix gateway(s). For a very -->
<!-- basic guide on Slurm usage, please see [this -->
<!-- guide](https://github.com/diku-dk/howto/blob/main/servers.md). -->
<!--  -->
<!-- Note that using Slurm can be tedious at first, and you risk suspension of your -->
<!-- access if you abuse the queue system, so use at your own caution. -->


## Other resources

### Futhark and CUDA

* We will use a basic subset of Futhark during the course. Futhark related documentation can be found at [Futhark's webpage](https://futhark-lang.org), in particular a [tutorial](https://futhark-book.readthedocs.io/en/latest/) and [user guide](https://futhark.readthedocs.io/en/stable/)

* [CUDA C Best Practices Guide](https://docs.nvidia.com/cuda/cuda-c-best-practices-guide/index.html) you may want to browse through this guide to see what offers. No need to read all of it closely.


### Other Related Books

* Some of the compiler transformations taught in the software track can be found
in this book [Optimizing Compilers for Modern Architectures. Randy Allen and Ken Kennedy, Morgan Kaufmann, 2001](https://www.elsevier.com/books/optimizing-compilers-for-modern-architectures/allen/978-0-08-051324-9), but you are not expected to buy it or read for the purpose of PMPH.

* Similarly, some course topics are further developed in this book [High-Performance Computing Paradigm and Infrastructure](https://www.wiley.com/en-dk/High+Performance+Computing%3A+Paradigm+and+Infrastructure-p-9780471732709), e.g., Chapters 3, 8 and 11, but again, you are not expected to buy it or read for the purpose of PMPH.

