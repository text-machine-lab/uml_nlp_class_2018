from pyfasttext import FastText


def main():
    model = FastText('model_text8.bin')

    target_word = 'dog'

    # get embedding
    target_word_embedding = model.get_numpy_vector(target_word)
    print('Target word:', target_word)
    print('Embedding shape:', target_word_embedding.shape)
    print('Embedding:', target_word_embedding[0:10], '...')

    # find closest words
    closest_words = model.nearest_neighbors(target_word, k=15)
    for word, similarity in closest_words:
        print('Word:', word, 'similarity:', similarity)


if __name__ == '__main__':
    main()
