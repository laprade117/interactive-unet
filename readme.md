
# Local setup

## First time run:
1. Copy the data into the data/image_volumes folder. All the volumes should be saved as .npy and have a 'uint8' datatype.
2. Create a new virtual environment using "python3 -m venv env"
3. Activate the environment with "source env/bin/activate"
4. Install requirements using "pip install -r requirements.txt"
5. If installing of requirements fails you might have to update pip with "pip install --upgrade pip" and then rerun "pip install -r requirements.txt"
6. Run the tool with "python unet.py -p 32544". "-p" is the port number. Any free port can be used. It will display a link that you can open similar to http://127.0.0.1:32544/. Open it in any web browser and the tool should work.

## All other runs:
Run commands 2 and 5.


# DTU Thinlinc Setup
This setup should work on DTU Thinlinc.

## First time run:
1. Copy the data into the data/image_volumes folder.
2. Open a terminal in the interactive-unet main folder.
3. Run the command "sxm2sh -X" to connect to the GPU's.
4. Run the command "module load python3/3.8.4"
5. Run the command "module load cudnn/v8.0.5.39-prod-cuda-11.0
6. Run the command "python3 -m venv env"
7. Run the command "source env/bin/activate"
8. Run "pip install -r requirements.txt"
9. Run the tool with "python unet.py -p 32544". "-p" is the port number. Any free port can be used. It will display a link that you can open similar to http://127.0.0.1:32544/. Open it in the web browser on Thinlinc.

## All other runs:
Run commands 3,4,5,7, and 9.


# DIKU Cluster Setup
I cannot get it to work on my end due to connection issues. It runs fine, but I cannot access the web interface so I can't actually use it. Theoretically, it should work by using the same method for remote jupyter notebook usage found at https://diku-dk.github.io/wiki/slurm-cluster.html. You may be able to figure it out, I just don't have much time right now to look into it.
