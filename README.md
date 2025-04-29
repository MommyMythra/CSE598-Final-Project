# Final Project (CSE 598) - MOPSO vs Off-The-Shelf MOGA

## Introduction

This is the final project for **CSE 598**, developed by **Joshua Elkins** and **Sangeet Ulhas**.  
We implemented a Python-based **Multi-Objective Particle Swarm Optimization (MOPSO)** algorithm and compared it against an off-the-shelf **Multi-Objective Genetic Algorithm (MOGA)** from the `pymoo` optimization library.  

The goal is to evaluate the performance of these two multi-objective optimization techniques on a set of benchmark functions commonly used in the field.

---

## Algorithms

- **MOPSO**: Multi-Objective Particle Swarm Optimization  
- **MOGA**: Multi-Objective Genetic Algorithm (from `pymoo`)

---

## Specific Variable Definitions
There are many variables used and defined throughout, but the major hyperparameters the user can control externally are:
- genCount: the number of generations
- runCount: the number of overarching runs

---

## Test Functions
1. **Zitzler–Deb–Thiele’s Function (ZDT)**
   - N = 2
   - 30 decision variables
   - 2 objectives 
   - 0 Constraints

2. **Kursawe Function**
   - 3 decision variables
   - 2 objectives
   - 0 Constraints

3. **Tanaka Function**
   - 2 decision variables
   - 2 objectives  
   - 2 Constraints

---

## Data Collected

From each run, the following information is collected:
1. The best pareto frontier
2. The associated solutions
3. The run time
4. The history of pareto frontiers

---

## How to Run

1. **Clone the repository**:
   ```bash
   git clone https://github.com/MommyMythra/CSE598-Final-Project.git
   ```

2. **Install the required libraries**
    ```bash
    pip install -r requirements.txt
    ```


3. **Navigate into the project folder**:
    ```bash
    cd CSE598-Final-Project
    ```
4. **Run the main script**:
    ```bash
    python3 main.py
    ```

5. **Input required Parameters**
    ```bash
    Input Number of Runs per Test Function: <Type a number here. I recommend 5-20>
    Input number of Generations per Run: <Type a number here. I recommend 50-100>
    ```
6. **Review results**
    A graphical window will show up containing plots and other relavent information
    The information will be saved to relevant PNGs and a textfile

---

## Remaining to Do List

1. ~Develop MOPSO in mopso.py~

2. ~Connect basic MOPSO to main function and validate it runs setup functions~

3. ~Display basic MOPSO using matplotlib~

4. ~Setup the Multi-Run framework (100 runs of 100 iterations for example)~

5. ~Collect all run information into a dataframe or simple list (Pareto History, Run time information, and best run)~

6. ~Process information and run calcs to finalize PLOTTING DATA~

7. ~Develop Mega-Graph Toolkit (A window that displays ALL revalent information)~

8. ~Validate Proper running on sufficient test and apply patches as needed~

9. ~Clean up code, fix naming conventions, add/clean comments~

10. ~Fix the Requirements~

11. ~Fixed Readme to be accurate and complete~
