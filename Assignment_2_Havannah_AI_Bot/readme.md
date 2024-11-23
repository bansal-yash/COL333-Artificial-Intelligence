## Creating environment

First install conda in your machine if not already present. Then run 

    conda create -n aia2 python=3.10 numpy tk
    
in your terminal to create the conda environment `aia2` with required packages. Do not install any other package in this environment as we will run your code with just these packages. You need to create the conda environment just once. 

Second time onwards, just activate the environment using the command
    
    conda activate aia2

and run the game command.

## Interacting with the Simulator

The following command will initiate a game of ExtendedHavannah between **agent_1** and **agent_2** with the initial state given at the **test_case_path**. The script can be invoked with several options described subsequently. The command should display the GUI showing the game. This total time available each player for the game is controlled by parameter `time`.

```python
python3 game.py {agent_1} {agent_2} --start_file {test_case_path} --time {total_time_in_seconds}
```

To create a random board specify board dimension (the number of cells in each edge) `dim` and run the following command:

```python
python3 game.py {agent_1} {agent_2} --dim 10 --time {total_time_in_seconds}
```

The following command will allow a **human agent** to play against the **AI agent.** The human agent (you) can click on a cell to play that move or type in a move as *"\<row num\> ,\<col num\>"*, where row and column numbers are 0 indexed:

```python
python3 game.py ai human --dim {board dimension} --time {total_time_in_seconds}
```

A simple **random agent** is provided in the starter code. The random agent simply picks its moves uniformly at random among the available ones. A game between an AI agent and the random agent can be initiated as follows:

```python
python3 game.py ai random --dim 5 --time 20
```

You might also see your bot play against itself:

```python
python3 game.py ai ai --start_file havannah/initial_states/size4.txt --time 20
```

Moves that are played on a blocked or out of window cell are considered **invalid moves**. If a player attempts to play an invalid move, the game simulator does not change the game state (i.e., the attempted move is skipped) and the turn switches to the next player. Note, that if at any point, if a player exhausts its total game time, it straight away loses and its opponent wins the game.
