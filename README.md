Bike_Sharing
============

Kaggle Competition



** Split the train data into train_,csv and validate_.csv **
** I have randomly put 10,000 records into train_.csv and remaining 887 records into the validation set ***

<b> Assumption : If you look at the way kaggle provided data for this competition, there is implied rule that you cannot use the future to predict the past.I have used the entire tranining set for learning the model and used that model to predict the test set. This violates the way the model would work in real world.  

TO-DO : Train models respecting the above implied rule
<b>


*Train the model 
python combined_model.py 0 data/train_.csv models/combined_model.p encoding/combined_model_encoding.p 

*Validate 
python combined_model.py 2 data/validate_.csv models/combined_model.p encoding/combined_model_encoding.p

*Predict on test set and upload to kaggle
python combined_model.py 1 data/test.csv models/combined_model.p encoding/combined_model_encoding.p output.csv 

Above gives a score of : 0.41449 on the test set.

**** I also trained to two separate models to predict casual and registered biker rentals separately. Just run the file split_model.py using parameters as above . This should give you a slighly better score on the test set : 0.41900. 





