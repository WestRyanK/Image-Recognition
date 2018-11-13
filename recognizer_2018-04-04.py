import cv2
import numpy as np
import glob
from scipy import spatial
from sklearn import svm
from sklearn.cluster import KMeans
from collections import Counter

class Recognizer:
    def __init__(self, categories_images, sample_count, vocab_size, verbose=False):
        self.__SIFT_DIM = 128
        self.__sift = cv2.xfeatures2d.SIFT_create()
        self.sample_count = sample_count
        self.vocab_size = vocab_size
        self.verbose = verbose

        vocab_images = []
        for catgry_img in categories_images:
            vocab_images.extend(catgry_img[1])

        self.__vocab = self.__build_vocab(vocab_images)
        self.__vocab_kd_tree = spatial.KDTree(self.__vocab)
        self.__train(categories_images)

    def __sample_descriptors(self, descriptor_list):
        M = self.sample_count * len(descriptor_list)
        descriptors_matrix = np.zeros((M, self.__SIFT_DIM))

        # Enumerate all images
        for i, img_descs in enumerate(descriptor_list):
            # Select a sample of descriptors from an image
            selections = np.random.permutation(len(img_descs))[0:self.sample_count]
            img_samples = ([img_descs[x] for x in selections])
            # Add the samples to the descriptors matrix
            for j, sample in enumerate(img_samples):
                descriptors_matrix[i * self.sample_count + j,:] = sample[:]
        return descriptors_matrix

    def __build_vocab(self, images):
        descriptor_list = self.__find_features(images)
        descriptors_matrix  = self.__sample_descriptors(descriptor_list)
    
        k_means = KMeans(n_clusters=self.vocab_size)
        k_means.fit(descriptors_matrix)
        vocab  = k_means.cluster_centers_
        return vocab

    def __find_words_in_vocab(self, descriptors):
        words = []
        for descriptor in descriptors:
            word = self.__vocab[self.__vocab_kd_tree.query(descriptor,1)[1]]
            words.append(word)
        return words
        
    def __find_features(self, images):
        kps = []
        descs = []
        if self.verbose:
            cv2.namedWindow("keypoints")
        for image in images:
            keypoints, descriptors = self.__sift.detectAndCompute(image, None)
            kps.append(keypoints)
            descs.append(descriptors)
            if self.verbose:
                im = cv2.drawKeypoints(image, keypoints, image, flags=cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)
                cv2.namedWindow
                cv2.imshow("keypoints", im)
                cv2.waitKey(200)
        if self.verbose:
            cv2.destroyWindow("keypoints")

        return descs
    
    def __get_categories_bags(self, categories_images):
        categories_bags = []
        for category, images in categories_images:
            bags_of_words = []
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
            for neg_img in neg_cat[1]:
                negative_bags.append(map(lambda ni: ni[1],neg_img))
        return negative_bags


    def __train(self, categories_images):
        self.__svms = []
        # categories_bags
        #   list of tuples of:
        #     0 - category
        #     1 - list of images:
        #         list of tuples of:
        #            0 - word
        #                array of ints for word feature
        #            1 - ratio
        categories_bags = self.__get_categories_bags(categories_images)
        for category, positive_bags in categories_bags:
            positive_bags2 = []
            for positive_bag in positive_bags:
                positive_bags2.append(map(lambda pb: pb[1], positive_bag))
            positive_bags = positive_bags2

            negative_bags = self.__build_negative_bags(categories_bags, category)
            train_matrix, label_matrix = self.__build_training_matrices(positive_bags, negative_bags)

            svm_model = svm.SVC(probability=True)
            svm_model.fit(train_matrix, label_matrix)
            self.__svms.append( (category, svm_model) )


    def __build_bag_of_words(self, image):
        kp, desc = self.__sift.detectAndCompute(image, None)
        words = self.__find_words_in_vocab(desc)
        word_tuples = map(lambda w: tuple(w), words)
        word_counts = dict(Counter(word_tuples))
        counts = map(lambda wc: word_counts[wc], word_counts)
        for word in map(lambda w: tuple(w), self.__vocab):
            if word not in word_counts:
                word_counts[word] = 0
        total = float(sum(word_counts.values()))
        proportional_word_counts = map(lambda wc: ( wc, word_counts[wc] / total), word_counts)
        bag = sorted(proportional_word_counts, key=lambda pwc: pwc[0])
        return bag

    def recognize(self, image):
        bag_of_words = self.__build_bag_of_words(image)
        bag_of_words = map(lambda pb: pb[1], bag_of_words)
        input = [bag_of_words]
        best_category = None
        best_proximity = None
        for category, svm_model in self.__svms:
#             a = svm_model.predict(input)
            b = svm_model.predict_proba(input)
            print "svm ", category
#             print "predict ", a
            print "probabi ", b



def load_images(path):
    image_paths = glob.glob(path)
    images = []
    for image_path in image_paths:
        pass
        images.append(cv2.imread(image_path))
    return images

catgry_imgs = []
ants = load_images("./train/ant/*")
elephants = load_images("./train/elephant/*")
flamingos = load_images("./train/flamingo/*")
catgry_imgs.append( ('ant', ants ) )
catgry_imgs.append( ('elephant', elephants))
catgry_imgs.append( ('flamingo', flamingos))

recognizer = Recognizer(
        categories_images=catgry_imgs,
        sample_count=200,
        vocab_size=200,
        verbose=False)
print "\n\n\nant 1"
recognizer.recognize(ants[1])
print "ant 2"
recognizer.recognize(ants[2])
print "ant 3"
recognizer.recognize(ants[3])

print "\n\n\nelephant 0"
recognizer.recognize(elephants[0])
print "elephant 1"
recognizer.recognize(elephants[1])
print "elephant 2"
recognizer.recognize(elephants[2])


print "\n\n\nflamingo 0"
recognizer.recognize(flamingos[0])
print "flamingo 1"
recognizer.recognize(flamingos[1])
print "flamingo 2"
recognizer.recognize(flamingos[2])
