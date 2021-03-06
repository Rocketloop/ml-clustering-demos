# :rocket:Rocketloop - Machine Learning Clustering Demos
This is a demo of a few unsupervised clustering algorithms in python.        
If you speak german we highly recommend reading our blog.          
You will find helpful information and insights about what the script does and how it works.        
Simply hit the following link:
https://rocketloop.de/machine-learning-clustering-in-python/
## The data
The dataset used is generated using points_generator.py. We already created the dataset and its saved under datasets/points.txt.
Feel free to use it (it's the default anyways).
## Running the script
### Running as Python3 application.
To run the script we recommend creating a virtual environment to install the required packages. 
Do so by typing `virtualenv venv` in your terminal in the directory where you want to install the environment.        
The next step will be to start the virtual environment with `source venv/bin/activate`.       
Once you are in the virtual environment enter `pip install -r requirements.txt` to install the required packages.
After the install you are good to run `python3 clusterer.py`.
For any changes of the data set used for training of the models you have to manually change the code.
To do so use an editor of your choice and navigate to the main function and edit the value for `data = pd.read_csv(your path)`.
Depending on your dataset you might have to declare categorical variables in the main function though you might have to do some more work preparing your data, therefore I don't recommend using categorical data unless you really know what you are doing.       
When you want to leave the virtual environment, simply type `deactivate` or quit your terminal.
### Running as Jupyter Notebook
To run the script in the Jupyter Notebook you must first install Jupyter.
To do so enter `pip install ipykernel` and `pip install ipython` in your terminal.     
After the installation change to the folder where these files are located in. 
Then simply enter `jupyter notebook`.
You will see Jupyter opening in your browser. All you need to do now is to select `clusterer.ipynb` and run all cells.
To quit Jupyter either close your terminal window or press Ctrl+c twice.
## Generate data with points_generator.py
If you want to create your own data to cluster you can use points_generator.py.      
All you need to do is specify the centers of the clusters. You can add or delete centers or just edit the existing ones.     
Next you will have to specify the amount of points for each cluster and how far they can be located relative to the center.     
Then all you need to do is to specify where the data should get saved by editin the file name in line 21.
