
# Keypoint detecton based onthis Based on Detectron2

<div align="center">
  <img src="output/Image0002.png" >
</div>
<br>

## Installatoin
 * First, install pytorch, then detectron2 first. Then migrate 'keypoint.py' and configs and data(20230104keypoints/2023109keyponts_test) from this repo to the installed detectron2.

* Install detectron2 from this repository (not recomended)

## Weight 

<a href="https://icsi01.s3.amazonaws.com/share/detectron2/output/model_final.pth?AWSAccessKeyId=ASIAQ64DGB6AHVHBHDE3&Expires=1682133224&x-amz-security-token=IQoJb3JpZ2luX2VjENv%2F%2F%2F%2F%2F%2F%2F%2F%2F%2FwEaDmFwLW5vcnRoZWFzdC0xIkcwRQIhAIExziDiG4r5Q%2FHmIBInks9RAdkIyLn%2Bwjae4BFw5z7PAiAU1KeuyWHyDPiHX36ZgJmLb8hMH3%2FYbHrYwi08QD%2BINSrJBQhUEAAaDDA2NjMxMDI0NjI3MiIM2bJo5wnlCI9ExmTCKqYFerAlgSHMbGUwyv%2FqevXe0M664W7tLUcy9DsQ6jq%2FF6CS7BvifpYIZJ1KrK93NtSuO3Z7Lg9NFaSMw4j9Vq0kYODCJA2Ult2cWOEGz28liR4aiA6lqUL2nH2SZ7FVYvWUBLG5thZ%2Fcel%2BV2xZYvz9iuHYVu1amRTxFtsRa%2BB%2FBnG%2FX%2B38ZmmJn%2BmzEk6XMty%2B7AopAywiI%2FxCPZGAb3hDEIXibpSFJ4pKv1HMrmuIUHA4U8bmiig6%2FP2LTd7UObHfhhc65YkChO6nz4Gl1NdO%2BUGcT0xh8RjMBOrvmNnuTAGV2KLRVUHsHtFndPPuxfxHzo8AX0SzSGqwHEs0IsMAj968Qh3GJHPTIJV1ZMitHGhJDyiSerDbD3a6ppfJnIYUA9N%2BxtP5NTRdcYih3rF3E%2BEHPHMv8gw1IFZko8GzVw8YjVUh9N%2BoOOteWkWYC0l3jaraCVLVW0glxWIfofmFqXcoogPW%2FifYNKXnusLGalGR6IoBX2UfyBkBgpwcY1krUWlftYVK1TfnGXv139FCyhNdTpaUSX75O%2FojpuY6nwjnbO8BeG6hrLuwzoSs3zBcvikSBiHfzzlBQ0Z9%2BsFLd7V67s86mTJ4XiXeANYKs9Q9SNY78JkVavxligAOTHjAfh8IpEbisZJ6vGc878uuIcgyWr3GSGOjQ34STWoRce7qSVmmzUDvXudlpXUBhsux8wYnSqCwU33Oi0BJaZXo1e727ktt2g9mGFAYP6BilMKiAeoJ5YBzwJv9%2B8FkWM2HIbgzNEE0ecSnb5Nqsz4%2BJzyhRiXq0Wut%2FINxvuc8UL%2FBfBJJ1Nx%2B8xM3LmVISPN2v9kdn%2FI98IM%2BuGrG1kl6BCl02dSBXzRCWPdhT%2ByZa8uzS3Q9zpu4tc9owWPEU291GNK7sYdWMJSX0p4GOrEB3kMAwwFAXkr85T7zC5UDLfcCiDOXI74ZtuKfBtLI3%2BKNdpWqc%2BIttWrZYeJ1OgFEzQlnqLvihFkA34YCwt4ti1ZvPuiUshFh8jUq%2FqyoWGxZomAqCSq%2FDhaNIjmiTxBLUkHEem3l4s5nNFQHzLDq0En1wWcv41E3v%2ByzsKrdcrmZYX5j6CrUF1yl%2BexNX%2FmFnZyNvQXYfGzB8e4JEFfa%2FDvNKO7dgXSAQ9baXdUxNzz2&Signature=4x%2FQPSI5XUB9Y06gisbvmbLKbew%3D" > Download model_final.pth (this weight is inital version)</a>


## Usage
All you have to read is keypoint.py which is used training and inference.

### Example : for prediction
```shell
  python keypoint.py --mode detect
```

### Example : for training
```shell
  python keypoint.py --mode train
```

## Ressume training
```shell
  # ressume training from outpoint/model_*****.pth, 
  # so you have to make symbolic link to model_final.pth
  ln -fs outpoint/model_*****.pth model_final.pth

  python keypoint.py --mode resume  # train model_final.pth
```

### other options
```python
 parser = argparse.ArgumentParser()
    parser.add_argument('--mode', type=str, default="detect", help='train,resume, detect')
    parser.add_argument('--basedir', type=str, default='/home/ssm-user/detectron2/20230104keypoints', help='base dir')
    parser.add_argument('--dataset', type=str, default='sp12', help='dataset name')
    parser.add_argument('--interactive', type=bool, default=True, help='interactive mode')
    parser.add_argument('--weight', type=str, default='model_final.pth', help='model weight')
    parser.add_argument('--output', type=str, default='output', help='output dir')
    parser.add_argument('--source', type=str, default="20230109keypoints_test/images", help='input source dir (test imaeges dir)')
    parser.add_argument('--sample', type=int, default=0, help='sample number')
    parser.add_argument('--config', type=str, default='COCO-Keypoints/keypoint_rcnn_R_101_FPN_3x.yaml', help='config file')
    parser.add_argument('--label', type=str, default="annotations/person_keypoints_default.json", help='label name relative to base dir')
    parser.add_argument('--image', type=str, default="annotations/images", help='training images dir')

```

## Result

'''shell
Prediction time: 1.4778940677642822
Predicted image saved to: output/Image0005.png
Predicting image: 20230109keypoints_test/images/Image0001.png
Prediction time: 1.1717393398284912
Predicted image saved to: output/Image0001.png
Predicting image: 20230109keypoints_test/images/Image0003.png
Prediction time: 1.220320463180542
Predicted image saved to: output/Image0003.png
Predicting image: 20230109keypoints_test/images/Image0002.png
Prediction time: 1.1189265251159668
Predicted image saved to: output/Image0002.png
Predicting image: 20230109keypoints_test/images/Image0008.png
Prediction time: 1.1416430473327637
Predicted image saved to: output/Image0008.png
Predicting image: 20230109keypoints_test/images/Image0006.png
Prediction time: 1.1780552864074707
Predicted image saved to: output/Image0006.png
Predicting image: 20230109keypoints_test/images/Image0009.png
Prediction time: 1.1545233726501465
Predicted image saved to: output/Image0009.png
Predicting image: 20230109keypoints_test/images/Image0004.png
Prediction time: 1.2353529930114746
Predicted image saved to: output/Image0004.png
Predicting image: 20230109keypoints_test/images/Image0007.png
Prediction time: 1.3687193393707275
Predicted image saved to: output/Image0007.png
Predicting image: 20230109keypoints_test/images/Image0010.png
Prediction time: 1.261420488357544
Predicted image saved to: output/Image0010.png
```

