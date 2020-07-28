# pg-cartpole

# What is this
play CartPole to 200 step with Policy Gradient using TensorFlow2.

# What is in
policygradient_cartpole_train.py : train models ,write these models in "model-save-path" folder;
policygradient_cartpole_test.py   : test models ,read models  from "model-save-path" folder ,test these model with several episodes ,write a result csv in "result-save-path"folder;
pg_cartpole_0300.h5                         ：a good model plays well;
pg_cartpole_test_result.csv            :  an example of result csv;


# How to use
## train
(1) make a new file to reserve the trained models  # default : ./
(2) python3 policygradient_cartpole_train.py    #default parameters is recommended

## test
python3 policygradient_cartpole_test.py

# other
(1) when testing , you can use render to show the cartpole,but it will be slow.
(2) the model will saved every 20 iteration(default),the example max-step curve shows below:








