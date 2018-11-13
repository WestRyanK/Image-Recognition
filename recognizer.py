import cv2
import numpy as np
import os
import glob
import time
from scipy import spatial
from sklearn import svm
from sklearn.cluster import KMeans
from collections import Counter

#%matplotlib inline
#np.set_printoptions(suppress=True)
#matplotlib.rcParams['figure.figsize'] = (4,4)

def plot2d(image, title="", max=None, min=0, cmap="gray"):
    plt.title(title)
    if max == None:
        plt.imshow(image, cmap=cmap, interpolation="none")
    else:
        plt.imshow(image, cmap=cmap, interpolation="none", vmin = min, vmax = max)
    plt.show()



class Recognizer:
    def __init__(self, categories_images, samples_per_image_count, vocab_size, max_images_per_category=25,  verbose=False):
        self.SIFT_DIM = 128
        self.sift = cv2.xfeatures2d.SIFT_create()
        self.samples_per_image_count = samples_per_image_count
        self.vocab_size = vocab_size
        self.max_images_per_category = max_images_per_category
        self.verbose = verbose

        categories_image = self.__limit_images_in_category(categories_images)

        vocab_images = []
        for catgry_img in categories_images:
            vocab_images.extend(catgry_img[1])

        self.vocab = self.__build_vocab(vocab_images)
        self.__alert("building vocab KD Tree...")
# I build a KDTree with the vocabulary words here 
# in order to quickly find which vocabulary word is closest for a feature
        self.vocab_kd_tree = spatial.KDTree(self.vocab)

        start_time = time.time()
        self.categories_bags = self.__get_categories_bags(categories_images)
        elapsed = time.time() - start_time 
        self.__alert("creating bags for all training images took " + str(elapsed) + " seconds")

        self.train(self.categories_bags)

    def __limit_images_in_category(self, categories_images):
        cat_imgs = []
        for i in range(len(categories_images)):
            if len(categories_images[i][1]) > self.max_images_per_category:
                new_cat_imgs = (categories_images[i][0], categories_images[i][1][:self.max_images_per_category])
                cat_imgs.append(new_cat_imgs)

        return cat_imgs




    def __sample_descriptors(self, descriptor_list):
        M = self.samples_per_image_count * len(descriptor_list)
        descriptors_matrix = np.zeros((M, self.SIFT_DIM))

        # Enumerate all images
        for i, img_descs in enumerate(descriptor_list):
            # Select a sample of descriptors from an image
            selections = np.random.permutation(len(img_descs))[0:self.samples_per_image_count]
            img_samples = ([img_descs[x] for x in selections])
            # Add the samples to the descriptors matrix
            for j, sample in enumerate(img_samples):
                descriptors_matrix[i * self.samples_per_image_count + j,:] = sample[:]
        return descriptors_matrix

    def __build_vocab(self, images):
        self.__alert("building vocab...")
        descriptor_list = self.__find_features(images)
        self.__alert("sampling descriptors...")
        descriptors_matrix  = self.__sample_descriptors(descriptor_list)
    
# Here I am clustering with KMeans to form a vocabulary using
# a sample of the SIFT descriptors from each training image
        self.__alert("clustering with K Means.\nThis may take a while...")
        start_time = time.time()
        k_means = KMeans(n_clusters=self.vocab_size)
        k_means.fit(descriptors_matrix)
        vocab  = k_means.cluster_centers_

        elapsed = time.time() - start_time 
        self.__alert("K means took " + str(elapsed) + " seconds")
        return vocab

    def __find_words_in_vocab(self, descriptors):
        words_list = []
        
        descriptor_sample_size = min(self.samples_per_image_count * 3, len(descriptors))
        for descriptor in descriptors[:descriptor_sample_size]:

            word = self.vocab[self.vocab_kd_tree.query(descriptor,1)[1]]
            words_list.append(word)
        return words_list
        
    def __find_features(self, images):
        descs = []
        self.__alert("finding features...")
        for image in images:
            _, descriptors = self.sift.detectAndCompute(image, None)
            descs.append(descriptors)

        return descs
    
    def __get_categories_bags(self, categories_images):
        self.__alert("finding bags of words for all training images...")
        categories_bags = []
        for category, images in categories_images:
            bags_of_words = []
            self.__alert("finding bags for " + category)
            for image in images:
                bag_of_words = self.__build_bag_of_words(image)
                bags_of_words.append(bag_of_words)
            categories_bags.append( (category, bags_of_words) )
        return categories_bags
        

    def __build_training_matrices(self, positive_bags, negative_bags):
        M = len(positive_bags) + len(negative_bags)
        train_matrix = np.zeros((M, self.vocab_size))
        label_matrix = np.zeros((M))

        for i, positive_bag in enumerate(positive_bags):
            train_matrix[i,:] = positive_bag
            label_matrix[i] = 1
            
        for i, negative_bag in enumerate(negative_bags):
            train_matrix[i + len(positive_bags),:] = negative_bag
            label_matrix[i + len(positive_bags)] = -1

        return train_matrix, label_matrix

    def __build_negative_bags(self, categories_bags, category):
        negative_categories = filter(lambda cat_bag: cat_bag[0] != category, categories_bags)
        negative_bags = []
        for neg_cat in negative_categories:
            negative_bags.extend(neg_cat[1])
        return negative_bags


    def train(self, categories_bags):
        self.__alert("training...")
        self.svms = []
        for category, positive_bags in categories_bags:
            self.__alert("building positive bags for " + category)
            positive_bags2 = []
            for positive_bag in positive_bags:
                positive_bags2.append( positive_bag)
            positive_bags = positive_bags2

            self.__alert("building negative bags for " + category)
            negative_bags = self.__build_negative_bags(categories_bags, category)
            self.__alert("building training matrices for " + category)
            train_matrix, label_matrix = self.__build_training_matrices(positive_bags, negative_bags)

# Here I train my SVM for classification of each category.
# Each SVM is trained on data within the category and
# data outside the category
            self.__alert("training " + category + " svm...")
            svm_model = svm.SVC(probability=True)
            svm_model.fit(train_matrix, label_matrix)
            self.svms.append( (category, svm_model) )
        self.__alert("training completed!")

    def __alert(self, msg):
        if self.verbose:
            print msg

# This method performs vector quantization. I am finding SIFT
# descriptors in the image and then finding which words those
# descriptors come from. Then I count each of the words and
# build a histogram from them.
    def __build_bag_of_words(self, image):
        kp, desc = self.sift.detectAndCompute(image, None)
        words = self.__find_words_in_vocab(desc)
        word_tuples = map(lambda w: tuple(w), words)
        word_counts = dict(Counter(word_tuples))
        counts = map(lambda wc: word_counts[wc], word_counts)
        for word in map(lambda w: tuple(w), self.vocab):
            if word not in word_counts:
                word_counts[word] = 0
        total = float(sum(word_counts.values()))
        proportional_word_counts = map(lambda wc: ( wc, word_counts[wc] / total), word_counts)
        bag = sorted(proportional_word_counts, key=lambda pwc: pwc[0])
        bag = map(lambda b: b[1], bag)
        return bag

    def recognize(self, image, output_stats=False):
        self.__alert("recognizing image...")
        bag_of_words = self.__build_bag_of_words(image)
        input = [bag_of_words]
        best_category = None
        best_proximity = 0
        stats = []
        for category, svm_model in self.svms:
            # This code right here uses my SVM to determine which
            # category the input image belongs to.
            # I test the probability of belonging to each category
            # given by each SVM and then return the category for the
            # SVM which gives the highest probability.
            svm_probability = svm_model.predict_proba(input)[0][1]
            if svm_probability > best_proximity:
                best_proximity = svm_probability
                best_category = category
            stats.append( (category, svm_probability) )

        if output_stats:
            return best_category, stats
        else:
            return best_category

def load_images(path):
    image_paths = glob.glob(path)
    images = []
    for image_path in image_paths:
        images.append(cv2.imread(image_path))
#         images.append(cv2.imread(image_path, cv2.IMREAD_GRAYSCALE ))
    return images

def load_categories(category_names):
    categories = []
    for name in category_names:
        imgs = load_images("./train/" + name + "/*")
        if len(imgs) == 0:
            raise Exception("bad category name: " + name)
            sys.exit(0)
        categories.append( (name, imgs ) )
    return categories

class bcolors:
    HEADER = '\033[95m'
    OKBLUE = '\033[94m'
    OKGREEN = '\033[92m'
    WARNING = '\033[93m'
    FAIL = '\033[91m'
    ENDC = '\033[0m'
    BOLD = '\033[1m'
    UNDERLINE = '\033[4m'

# This method runs a test using our recognizer to determine 
# whether an image belongs to the correct_category.
# Probabilities for each category are displayed
def test_image(image, correct_category):
    is_correct = False
    output_img = np.copy(image)
    recognized_category, stats = recognizer.recognize(output_img, output_stats=True)

    color = (0,0,255)
    if recognized_category == correct_category:
        is_correct = True
        color = (0, 255, 0)

    confidence = max(map(lambda s: s[1], stats))
    confidence = '{:.1%}'.format(confidence)

    cv2.putText(output_img, recognized_category, (5,30), cv2.FONT_HERSHEY_PLAIN, 1.5, color, 3)
    cv2.putText(output_img, str(confidence), (5,60), cv2.FONT_HERSHEY_PLAIN, 1.5, color, 3)
    output_img = cv2.cvtColor(output_img, cv2.COLOR_BGR2RGB)
    plot2d(output_img, confidence)

    for stat in stats:
        color = bcolors.ENDC
        if stat[0] == recognized_category:
            if recognized_category == correct_category:
                color = bcolors.OKGREEN
            else:
                color = bcolors.FAIL
        print color,'{:<15}{:>6.1%}'.format(stat[0], stat[1])
    return is_correct, output_img
    
# Runs test_image for a bunch of sample data from all of the
# categories that are specified.
def test_recognition(categories, test_samples_per_category):
    correct_matches = 0
    total_tests = len(categories) * test_samples_per_category
    results_imgs = []
    for category, imgs in categories:
        print bcolors.BOLD, '{:-^40}'.format("Category " + category), bcolors.ENDC
        for i in range(test_samples_per_category):
            is_correct, output_img = test_image(imgs[len(imgs) - i - 1], category)
            results_imgs.append(output_img)
            if is_correct:
                correct_matches += 1
                
    print "\n\n\n\n"
    print "accuracy: ", correct_matches, " of ", total_tests
    percent_accuracy = correct_matches / float(total_tests)
    print '{:.1%}'.format(percent_accuracy)
    return results_imgs


category_names = [
    "apricot_tree", "popcorn", "revolver"]
#          "metronome",   "menorah",      "sunflower",    "strawberry",   "snoopy",
#          "brain",       "buddha",       "soccer_ball",  "grand_piano",  "revolver",
#          "yin_yang",    "stop_sign",    "pyramid",      "nautilus",     "scissors",
#          "schooner",    "ceiling_fan",   "laptop",       "inline_skate", "dollar_bill"]

print "loading image files..."
categories = load_categories(category_names)

recognizer = Recognizer(
        categories_images=categories,
        samples_per_image_count=80,
        vocab_size=125, max_images_per_category=15,
        verbose=True)


recognizer.train(recognizer.categories_bags)



TEST_IMAGES_PER_CATEGORY = 6
results_imgs = test_recognition(categories, TEST_IMAGES_PER_CATEGORY)
