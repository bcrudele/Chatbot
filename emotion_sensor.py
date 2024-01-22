import string

### Notes for later:
# - expand training data
# - remove filler words like "the"
# - create case for words that are not in training data

def find_emotion(emotion_dict):
    ### CAN BE OPTIMIZED
    # written by ChatGPT

    # Check if 'total' key is present
    #print(emotion_dict['total'])
    if 'total' in emotion_dict and emotion_dict['total'] > 0:
        # Calculate the percentage for each emotion
        percentages = {emo: emotion_dict[emo] / emotion_dict['total'] * 100 for emo in emotion_dict.keys() if emo != 'total'}
        return percentages
    else:
        # Return None if 'total' is not present or is zero
        # Calculate the total sum of all emotion counts
        total_emotion_sum = sum(emotion_dict[emo] for emo in emotion_dict.keys() if emo != 'total')

        # Calculate the percentage for each emotion based on the total sum
        percentages = {emo: emotion_dict[emo] / total_emotion_sum * 100 for emo in emotion_dict.keys() if emo != 'total'}
        return percentages
    


def remove_punctuation(input_string):
    translation_table = str.maketrans("", "", string.punctuation)
    return input_string.translate(translation_table)


def word_frequency(data, label):
    words = dict()
   
    for i in range(len(data)):
        #words_list = data[i].lower().split()  # creates the list for words
        words_list = remove_punctuation(data[i].lower()).split()

        for word in words_list:
            if word not in words:             # for first word found
                words[word] = {'total': 1, 'sadness': 0, 'anger': 0, 'love': 0, 'surprise': 0, 'fear': 0, 'happy': 0, 'content': 0}

            else: 
                words[word]['total'] += 1

            if label[i] ==   'sadness':         
                words[word]['sadness'] += 1

            elif label[i] == 'anger':         
                words[word]['anger'] += 1

            elif label[i] == 'love':        
                words[word]['love'] += 1

            elif label[i] == 'surprise':        
                words[word]['surprise'] += 1

            elif label[i] == 'fear':         
                words[word]['fear'] += 1

            elif label[i] == 'happy':         
                words[word]['happy'] += 1

            else:
                words[word]['content'] += 1

    return words

train_file = 'train_data/emotion_dataset.csv'

with open(train_file, 'r') as f:
    train_lines = f.readlines()

train_data, train_label = [], []

for line in train_lines:
    split = line.rsplit(',', maxsplit=1)    # splits the line by the last comma
    train_data.append(split[0].strip())     # get text/phrase
    train_label.append(split[1].strip())    # get emotion

words =  word_frequency(train_data, train_label)

### FORMATTED OUTPUT

#test_word = 'surely'
#print(words[test_word])

#print('Top 20 angry words')
#sorted_words = sorted(words.items(), key=lambda x: (x[1]['anger'] / x[1]['total'], x[1]['total']), reverse=True)[:20]
#for word, stats in sorted_words:
#    print(f"Word: {word:<15} | Anger Ratio: {stats['anger']/stats['total']:.2%} | Total: {stats['total']}")

###
#max_emotion, max_count = find_max_emotion(words["you"])
#print(max_emotion, max_count)

test_phrase = [
    "in a sea of sorrow  sad often depression strikes bad  cry  darkness surrounds "
]

threshold = 20

for i in range(len(test_phrase)):

    phrase_score = {
        'sadness': 0,
        'anger': 0,
        'love': 0,
        'surprise': 0,
        'fear': 0,
        'happy': 0,
        'content': 0
    }
    # Convert the phrase to lowercase and split it into words
    words_list = remove_punctuation(test_phrase[i].lower()).split()

    for phrase_word in words_list:
        emotion_percentages = find_emotion(words[phrase_word])
        print(phrase_word, emotion_percentages)

        # Check if emotion_percentages is not None
        if emotion_percentages:
            # Increment the corresponding emotion count based on the highest percentage
            max_emotion = max(emotion_percentages, key=emotion_percentages.get)

            if emotion_percentages[max_emotion] > threshold:
                phrase_score[max_emotion] += 1

    print(phrase_score)
    phrase_percent = find_emotion(phrase_score)
    print(phrase_percent)
    ### SOMETHING WRONG WITH LINE BELOW
    print(max(phrase_percent,key=emotion_percentages.get ))