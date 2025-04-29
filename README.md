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

*To be determined*

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

*To be determined*

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
5. **Review results**
    A graphical window will show up containing plots and other relavent information

---

## Remaining to Do List

1. ~Develop MOPSO in mopso.py~

2. ~Connect basic MOPSO to main function and validate it runs setup functions~

3. ~Display basic MOPSO using matplotlib~

4. ~Setup the Multi-Run framework (100 runs of 100 iterations for example)~

5. Collect all run information into a dataframe or simple list (Pareto History, Run time information, and best run)

6. Develop Mega-Graph Toolkit (A window that displays ALL revalent information)
