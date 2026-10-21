# Presentation Notes: Fruit & Vegetable Freshness Detection System

A 3–5 minute talk, followed by likely teacher questions with short answers.
Wherever a number is marked **[see results]**, read it from `results/metrics.json` /
README → Results. Only quote numbers the evaluation actually produced.

---

## Part A: the 3–5 minute talk

### 1. Problem (about 20 s)
"Shops, warehouses and homes throw away a lot of fruit and vegetables, and spoiled
produce sometimes reaches customers because checking is done by eye, which is slow and
inconsistent. We wanted a computer to look at a photo and say whether an item is fresh
or spoiled."

### 2. Solution (about 20 s)
"You upload a photo in our web app. A trained deep-learning model answers, for
example: *Fruit/Vegetable: Banana, Freshness: SPOILED, Confidence: …%*. It supports
14 fruits and vegetables."

### 3. Dataset (about 30 s)
"We used the public Kaggle dataset *Fruit and Vegetable Disease (Healthy vs Rotten)*,
CC0 licence, about 29,000 images in 28 folders: 14 produce types, each healthy or
rotten. Nine are fruits (apple, banana, grape, guava, jujube, mango, orange,
pomegranate, strawberry) and five are vegetables (bell pepper, carrot, cucumber,
potato, tomato). It is compiled from several sources, so we removed duplicate images
first." *(Show `DATASET.md` and `results/class_distribution.png`.)*

### 4. Machine learning (about 15 s)
"This is **supervised image classification**: the input is an image, the output is
one of 28 labels such as *Apple__Rotten*. Produce type and freshness are both read
from that label."

### 5. CNN (about 20 s)
"We use a **convolutional neural network**. Convolution filters slide over the image
and detect local patterns. Early layers find edges and colours, deeper layers find
textures such as mould, wrinkles and dark spots, and the last layers combine them into
a decision. A CNN reuses the same filter everywhere, so a rotten spot is recognised
wherever it is in the picture."

### 6. Transfer learning (about 25 s)
"Training a CNN from scratch needs millions of images. Instead we took
**MobileNetV2 already trained on ImageNet** (1.2 million photos), kept its feature
extractor and replaced only the final layer with a new 28-class layer. We first
trained only that new layer, then **fine-tuned** the last few blocks with a small
learning rate. MobileNetV2 is small and fast (3.5 million parameters), so it trains in
minutes on a laptop GPU and could run on a phone."

### 7. Preprocessing (about 25 s)
"Before training, every image is checked for corruption, rotated upright, converted
to RGB and resized to 224×224. Pixels are normalised with the ImageNet mean and
standard deviation. We found duplicates with a perceptual hash, so the same photo
cannot sit in both the training and the test set (that would be data leakage).
Finally a stratified split gives 70% training, 15% validation and 15% test."
*(Show `src/data_preprocessing.py` → `main()`.)*

### 8. Data augmentation (about 15 s)
"During training every image is randomly flipped, rotated up to 15°, shifted, zoomed
and brightened a little, so the model sees a new version each epoch and does not
memorise. We do **not** change colours, because colour is what shows spoilage."
*(Show `src/dataset.py` → `get_train_transform()`.)*

### 9. Training (about 20 s)
"The model predicts, the cross-entropy loss measures how wrong it was,
backpropagation computes gradients, and the Adam optimizer updates the weights, one
batch of 32 images at a time. We save the best model on the validation set, reduce
the learning rate when progress stalls, and stop early when it no longer improves.
Training ran on the laptop's RTX 5050 GPU." *(Only say GPU if `results/training_summary.json` shows `"device": "cuda"`.)*
*(Show `src/train.py` → `run_epoch()` and `results/training_accuracy.png`.)*

### 10. Evaluation (about 25 s)
"We tested on the 15% of images the model never saw. Test accuracy was
**[see results]**, macro F1 **[see results]**. The confusion matrix shows which
classes get mixed up." *(Show `results/confusion_matrix.png` and
`results/classification_report.txt`.)*

### 11. Prediction (about 20 s, live demo)
"When you upload a photo, the server checks it is a real image, applies exactly the
same resizing and normalisation as in training, runs the model, turns the scores into
probabilities with softmax and shows the top class. If the top probability is below
70%, it warns that the image may not belong to the supported categories."
*(Upload a sample image in the web app.)*

### 12. Limitations (about 20 s)
"The model only knows 14 produce types; anything else is forced into one of them. It
only sees the outside, so it cannot detect internal rot or tell if food is safe to
eat. The dataset comes from internet images, so real shop photos with clutter and odd
lighting may be harder."

---

## Part B: likely teacher questions and answers

**1. Why did you choose this project?**
Food waste and food quality are real problems, spoilage is mostly visible, and the
project covers the whole ML workflow: data, preprocessing, training, evaluation and
deployment in an app.

**2. Why is this a machine-learning problem?**
No simple rule can describe "rotten" for 14 very different items: a brown banana, a
mouldy strawberry and a wrinkled potato look completely different. Instead of
hand-writing rules, we let a model **learn** the patterns from thousands of labelled examples.

**3. Why image classification?**
Each image contains one item and we want one label per image, such as
"Rotten Banana". That is exactly image classification. Finding several items in one
picture would be object detection, which is a future improvement.

**4. Why a CNN?**
CNNs are built for images. Convolution filters detect local patterns (edges → textures
→ parts), share weights across the image (fewer parameters, position-independent) and
keep the 2D structure. A normal fully-connected network on 150,528 raw pixels would
need far more parameters and would ignore the spatial structure.

**5. What is transfer learning?**
Reusing a model trained on a large dataset (ImageNet) for a new task. The early and
middle layers already detect general visual features, so we only need to train a new
final layer (and fine-tune the top layers) on our smaller dataset. It needs less data,
less time and usually gives better accuracy than training from scratch.

**6. Why MobileNetV2?**
It is lightweight (3.5 M parameters, 0.3 GFLOPs per image, against 25.6 M and 4.1
GFLOPs for ResNet-50), trains fast on an 8 GB laptop GPU, can run on phones, and is
accurate enough for this task. EfficientNet-B0 was the alternative (slightly better on
ImageNet, slightly slower). Our code supports it with `--arch efficientnet_b0`.

**7. What dataset did you use?**
"Fruit and Vegetable Disease (Healthy vs Rotten)" from Kaggle by Muhammad Subhan,
CC0 public-domain licence. It is compiled from other public datasets.

**8. How many images are in the dataset?**
Kaggle lists 29,291 image files. After removing corrupted files and duplicates we used
**[see DATASET.md section 4]** images.

**9. How many classes are there?**
28 classes: 14 produce types × {healthy, rotten}. That is 9 fruits and 5 vegetables.

**10. How did you split the dataset?**
70% training, 15% validation, 15% test, stratified per class (each class keeps the
same proportions), with a fixed random seed. Look-alike images are always kept in the
same split to prevent leakage. The test set is used only once, at the end.

**11. What is overfitting?**
When the model memorises the training images (including noise) instead of learning
general patterns. You see it when training accuracy keeps rising but validation
accuracy stops improving or falls.

**12. How did you prevent overfitting?**
Transfer learning (few parameters trained at first), data augmentation, dropout
(0.3) in the classifier, a small fine-tuning learning rate, early stopping on
validation loss, keeping the best checkpoint, and removing duplicate images.

**13. What is data augmentation?**
Creating slightly changed copies of the training images on the fly (flip, small
rotation, shift, zoom, brightness), so the model learns that these changes do not
change the label. It effectively enlarges the dataset. We do not change colour,
because colour carries the spoilage information.

**14. What is normalisation?**
Rescaling pixel values to a standard range. We divide by 255 (→ 0–1), then subtract
the ImageNet mean and divide by the ImageNet standard deviation per colour channel,
because the pretrained network expects inputs on that scale. It also makes training
more stable.

**15. What is an epoch?**
One complete pass through all training images. We trained for up to 5 epochs
(head only) plus up to 15 epochs (fine-tuning), with early stopping.

**16. What is batch size?**
The number of images processed together before the weights are updated. We used 32:
large enough for stable gradients and fast GPU use, small enough for the GPU memory.

**17. What is the learning rate?**
How big a step the optimizer takes when updating the weights. Too large and training
jumps around or diverges; too small and it learns very slowly. We used 0.001 for the
new head and 0.0001 for fine-tuning, and reduce it automatically when validation loss stops improving.

**18. What loss function did you use?**
Categorical **cross-entropy**. It is −log(probability given to the correct class):
small when the model is confident and correct, large when it is confident and wrong.
It is class-weighted, so mistakes on rare classes (200 images) count more.

**19. What optimizer did you use?**
**Adam**. It is gradient descent that keeps a running average of past gradients
(momentum) and adapts the step size for each weight. It works well with little tuning.

**20. What is accuracy?**
Correct predictions ÷ all predictions on the test set. It is easy to understand but
can hide poor performance on small classes, which is why we also report macro F1.

**21. What are precision and recall?**
For a class such as "Rotten Apple":
*Precision* = of everything the model called rotten apple, the fraction that really
was (few false alarms).
*Recall* = of all real rotten apples, the fraction the model found (few misses).

**22. What is the F1-score?**
The harmonic mean of precision and recall: 2·P·R / (P + R). It is high only when both
are high. **Macro F1** averages it over all 28 classes equally.

**23. What does the confusion matrix tell you?**
Rows are the true classes, columns the predicted classes. The diagonal shows correct
predictions and everything else shows mistakes. It tells us which classes get confused,
for example fresh vs rotten of the same fruit, or two similar-looking fruits.

**24. What happens if the uploaded image is not in the dataset (e.g. a lemon)?**
The model still has to choose one of its 28 classes, because softmax always sums to
100%. Often the top probability is low, and then the app shows a *low confidence*
warning. But a model can also be wrongly confident on unknown objects, so we clearly
list the supported classes and do not claim it recognises anything else.

**25. Can this system detect internal spoilage?**
No. It only sees the outside surface in the photo. Internal rot, bruising under the
skin or contamination needs other sensors, such as near-infrared or hyperspectral
imaging, gas (ethylene/CO₂) sensors, or cutting the item open.

**26. Can it detect every fruit and vegetable?**
No. Only the 14 types in the dataset: apple, banana, grape, guava, jujube, mango,
orange, pomegranate, strawberry, bell pepper, carrot, cucumber, potato and tomato.

**27. What are the limitations?**
Only 14 types and 2 freshness levels; external appearance only; mostly single items on
simple backgrounds; the dataset is compiled from internet sources, with class imbalance
and quality differences between classes (some rotten classes have much smaller images);
test accuracy comes from images of the same sources, so real-world accuracy would
probably be lower; softmax confidence is not a reliable "unknown object" detector.

**28. How could the system be improved?**
More and more varied data (real shop photos), more freshness levels or shelf-life
prediction, object detection for crates or conveyor belts, an explicit "unknown" class,
Grad-CAM heat maps to explain decisions, mobile/edge deployment, real-time camera input,
and combining images with gas, temperature or hyperspectral sensors.

**29. Why didn't you use traditional machine learning (e.g. SVM on colour histograms)?**
Traditional ML needs hand-crafted features such as colour histograms or texture
descriptors. Spoilage looks very different for each produce type and depends on
lighting, so hand-made features are fragile. A CNN learns the features itself from the
data, and with transfer learning it does so with relatively few images. Deep learning
is the standard, best-performing approach for image recognition.

**30. How would you deploy this in a supermarket or agricultural setting?**
Put a camera over a conveyor belt or sorting table, with fixed lighting and a plain
background. Run an object detector to find each item, then this classifier on each
crop, on an edge device (e.g. NVIDIA Jetson) with a model exported to ONNX/TensorRT.
Low-confidence items go to a human for checking. Staff photos of mistakes are collected
to retrain the model regularly, and accuracy is monitored over time. For shoppers or
farmers, a mobile app with the model on the phone.

---

## Part C: demo checklist (what to open, in order)

1. **VS Code → Explorer:** show the folder structure (README.md → *Project structure*).
2. **Dataset:** `DATASET.md`, `dataset/raw/…` (a few class folders), `results/class_distribution.png`.
3. **Preprocessing:** `src/data_preprocessing.py` → `process_one_image()`,
   `remove_duplicates_and_group()`, `split_dataset()`; `src/utils.py` → `load_and_resize()`.
4. **Model:** `src/model.py` → `build_model()`, `freeze_backbone()`, `unfreeze_last_blocks()`.
5. **Training:** `src/train.py` → `run_epoch()` (forward, loss, backward, step) and
   `train_phase()` (checkpoint, early stopping, LR reduction); `results/training_log.txt`.
6. **Evaluation:** `results/training_accuracy.png`, `results/training_loss.png`,
   `results/confusion_matrix.png`, `results/classification_report.txt`.
7. **Saved model:** `models/fruit_vegetable_freshness_model.pt` and `models/class_labels.json`.
8. **Start Flask:** in the terminal, `.\run.bat app`, then open http://127.0.0.1:5000.
9. **Upload an image:** pick a file from `sample_images/` (or click a thumbnail) and press **Detect Freshness**.
10. **Show the prediction**, then (optional) run `.\run.bat predict sample_images\<file>` in the terminal to show the same result from the command line.
