# Annotate images by selecting best looking model

~~~
opencv == 3.4.14.51(whatever version that will not cause X11 window error while imshow),  
numpy,  
pillow  
~~~
packages are required

in the step2 folder, you will be given the worklist containing the file ipaths that you have to annotate.  
worklist file should be in  
~~~
{worker_name}_job.csv
~~~
given the worklist file above, execute
~~~
python determine_best_models.py --worker_name {worker_name} --cams_path {PATH_TO_GRADCAM_RESULTS}
~~~
here, when gradcam results lie in 

~~~
ROOT_PATH
|
--step1
  |
  --gradcams
    |
    --model1
    --model2
    --model3
  |
  --images
  ...
|
--step2
|
...
~~~
then, type
~~~
python determine_best_models.py --worker_name {worker_name} --cams_path ../step1/gradcams
~~~
