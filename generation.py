import sys
import time
import re
from pathlib import Path
import math
from scipy import spatial
import numpy as np

# Récupère la table associative 
def open_table():
    data = ""
    with open("TableAssociative") as file:
        data = file.read()
    return data

# Génère un dictionnaire avec le pos_tag en clé et les mots associés en valeur
def generate_table_dict(data):
    lines = data.split("\n")
    dico = {}

    i=0
    for x in lines:
        lines[i] = re.sub('\t', " ", x)
        lines[i] = lines[i].split()
        dico[lines[i][0]] = lines[i][1:]
        i=i+1
    return dico

# Récupère les embeddings
def open_embeddings():
    data = ""
    with open("embeddings-Fr.txt") as file:
        data = file.read()
    return data

# Génère un dictionnaire des embeddings avec le mot en clé et son vecteur en valeur
def generate_embeddings_dict(data):
    lines = data.split("\n")

    dico2={}

    i=0
    for x in lines:
        lines[i] = re.sub('[\s[\],]', " ", x)
        lines[i] = lines[i].split()
        dico2[lines[i][0]] = lines[i][1:]
        i=i+1
    return dico2

print("Generating table dict..")
start = time.time()
table = open_table()
table_dict = generate_table_dict(table)
print("Generated table dict in %s seconds" % (time.time() - start))

print("Generating embed dict..")
start = time.time()
embed = open_embeddings()
embed_dict = generate_embeddings_dict(embed)
print("Generated embed dict in %s seconds" % (time.time() - start))

#Récupère les templates pour traitement
def open_template():
    data = ""
    # with open("templates_basiques", 'r') as file:
    #     data = file.read()
    #     data += "\n"
    # with open("templates_basiques_2", 'r') as file:
    #     data += file.read()
    with open("templates_eval", 'r') as file:
         data = file.read()
    return data.split("\n")

# Regex pour extraire les pos_tags & les deux mots suivants de chaque ligne
def reg_template(text):
    text = re.findall("[*](\S*)\s", text)
    return text

# Récupère les pos_tags & les deux mots suivants de chaque ligne d'un fichier de templates
def get_tags(text):
    tags = []
    for line in text:
        line = reg_template(line)
        tags.append(line)
    return tags

# Retourne les pos_tags & les deux mots suivants d'une seule ligne
def get_line_tags(line):
    return reg_template(line)

# Retourne une liste propre des 
def get_clear_tags(tags):
    clear_tags = []
    for line in tags:
        line_tags = []
        for element in line:
            line_tags.append(element.split("/"))
        clear_tags.append(line_tags)
    return clear_tags

def get_clear_line_tags(tags):
    clear_tags = []
    for element in tags:
        clear_tags.append(element.split("/"))
    return clear_tags

print("Extracting base templates 1 and 2..")
start = time.time()
text = open_template()
# print(text)
# tags = get_tags(text)
# clear_tags = get_clear_tags(tags)
print("Extracted templates in %s seconds" % (time.time() - start))
# print(clear_tags[1][1])



###################################
# Calculs de distances différents #
###################################

def euclidian_distance(point_a, point_b):
    distance = 0
    for i in range(len(point_a)):
        distance += (float(point_a[i])-float(point_b[i]))**2
    return math.sqrt(distance)

def cosine_similarity(point_a, point_b):
    return 1 - spatial.distance.cosine(point_a, point_b)

def chebyshev_distance(point_a, point_b):
    return spatial.distance.chebyshev(point_a, point_b)

def braycurtis_distance(point_a, point_b):
    return spatial.distance.braycurtis(point_a, point_b)

def canberra_distance(point_a, point_b):
    return spatial.distance.canberra(point_a, point_b)

# Si le pos_tag contient la lettre F, on retourne le mot de gauche (féminin) a comparer, sinon on retourne le mot de droite (masculin)
def return_slash_word(pos_tag,words):
    for char in pos_tag:
        if char == "F":
            return words[0]
    return words[1]

def get_closest_euclidian(pos_tag, query, ban_words, slash_words):
    dist = float("inf")
    word = ""
    furthest_word = return_slash_word(pos_tag, slash_words)
    if furthest_word in embed_dict:
        dist = -1
    for item in table_dict[pos_tag]:
        if(item in embed_dict and item not in ban_words):
            if furthest_word in embed_dict:
                min_max = euclidian_distance(embed_dict[item], embed_dict[furthest_word]) - euclidian_distance(embed_dict[item], embed_dict[query])
                if(min_max > dist):
                    dist = min_max
                    word = item
            else:
                if(euclidian_distance(embed_dict[item], embed_dict[query]) < dist):
                    dist = euclidian_distance(embed_dict[item], embed_dict[query])
                    word = item
    return word

def get_closest_cosine(pos_tag, query, ban_words, slash_words):
    dist = -1
    word = ""
    furthest_word = return_slash_word(pos_tag, slash_words)
    for item in table_dict[pos_tag]:
        if(item in embed_dict and item not in ban_words):
            if furthest_word in embed_dict:
                min_max = cosine_similarity(list(np.float_(embed_dict[item])), list(np.float_(embed_dict[furthest_word]))) - cosine_similarity(list(np.float_(embed_dict[item])), list(np.float_(embed_dict[query])))
                if(dist != min(min_max, dist, key=abs)):
                    dist = min_max
                    word = item
            else:
                min_max = cosine_similarity(list(np.float_(embed_dict[item])), list(np.float_(embed_dict[query])))
                if(dist != min(min_max, dist, key=abs)):
                    dist = min_max
                    word = item
    return word

def get_closest_chebyshev(pos_tag, query, ban_words, slash_words):
    dist = float("inf")
    word = ""
    furthest_word = return_slash_word(pos_tag, slash_words)
    if furthest_word in embed_dict:
        dist = -1
    for item in table_dict[pos_tag]:
        if(item in embed_dict and item not in ban_words):
            if furthest_word in embed_dict:
                min_max = chebyshev_distance(list(np.float_(embed_dict[item])), list(np.float_(embed_dict[furthest_word]))) - chebyshev_distance(list(np.float_(embed_dict[item])), list(np.float_(embed_dict[query])))
                if(min_max > dist):
                    dist = min_max
                    word = item
            else:
                if(chebyshev_distance(list(np.float_(embed_dict[item])), list(np.float_(embed_dict[query]))) < dist):
                    dist = chebyshev_distance(list(np.float_(embed_dict[item])), list(np.float_(embed_dict[query])))
                    word = item
    return word

def get_closest_braycurtis(pos_tag, query, ban_words, slash_words):
    dist = float("inf")
    word = ""
    furthest_word = return_slash_word(pos_tag, slash_words)
    if furthest_word in embed_dict:
        dist = -1
    for item in table_dict[pos_tag]:
        if(item in embed_dict and item not in ban_words):
            if furthest_word in embed_dict:
                min_max = braycurtis_distance(list(np.float_(embed_dict[item])), list(np.float_(embed_dict[furthest_word]))) - braycurtis_distance(list(np.float_(embed_dict[item])), list(np.float_(embed_dict[query])))
                if(min_max > dist):
                    dist = min_max
                    word = item
            else:
                if(braycurtis_distance(list(np.float_(embed_dict[item])), list(np.float_(embed_dict[query]))) < dist):
                    dist = braycurtis_distance(list(np.float_(embed_dict[item])), list(np.float_(embed_dict[query])))
                    word = item
    return word

def get_closest_canberra(pos_tag, query, ban_words, slash_words):
    dist = float("inf")
    word = ""
    furthest_word = return_slash_word(pos_tag, slash_words)
    if furthest_word in embed_dict:
        dist = -1
    for item in table_dict[pos_tag]:
        if(item in embed_dict and item not in ban_words):
            if furthest_word in embed_dict:
                min_max = canberra_distance(list(np.float_(embed_dict[item])), list(np.float_(embed_dict[furthest_word]))) - canberra_distance(list(np.float_(embed_dict[item])), list(np.float_(embed_dict[query])))
                if(min_max > dist):
                    dist = min_max
                    word = item
            else:
                if(canberra_distance(list(np.float_(embed_dict[item])), list(np.float_(embed_dict[query]))) < dist):
                    dist = canberra_distance(list(np.float_(embed_dict[item])), list(np.float_(embed_dict[query])))
                    word = item
    return word

# Retourne true si la query est un des mots de la liste
def is_in_tags(query, liste):
    for x in liste:
        for element in x:
            if(query == element):
                return True
    return False


# Main
print("")
print("Donner la query (thème) :")
query = input()
for line in text:

    print("")
    print("#########")
    print("")

    ban_list = {}
    ban_words = {}

    tags = get_line_tags(line)
    clear_tags = get_clear_line_tags(tags)

    while True:

        # print(re.sub("[*](\S*)\s", "XXX ", line))
        # print("")
        # print("Donner la query (thème) :")
        # query = input()

        # Si la query n'est pas dans les embeddings (donc pas comparable) ou fait partie des mots donnés avec le pos_tag dans les SGP, on ne process pas et on passe a la SGP suivante
        if (query not in embed_dict.keys() or is_in_tags(query,clear_tags)):
            print("")
            print("//// Query incorrecte, réessayez avec une autre /////")
            print("")
            break
        
        else:
            words = []
            for i in range(len(clear_tags)):
                # On check si on a déjà des ban_words pour le pos_tag 
                if(clear_tags[i][0] in ban_words and clear_tags[i][0] in ban_list):
                    ban_list[clear_tags[i][0]] = ban_words[clear_tags[i][0]] + clear_tags[i][1:]
                else:
                    ban_list[clear_tags[i][0]] = clear_tags[i][1:]

                # On récupère aussi les deux mots donnés avec le pos_tag dans la SGP 
                slash_words = clear_tags[i][1:]

                # Si on a déjà process des mots ( = si on ne process pas le premier pos_tag de la phrase)
                if(len(words) != 0):
                    # word = get_closest_euclidian(clear_tags[i][0], query, ban_list[clear_tags[i][0]] + words, slash_words)
                    # word = get_closest_cosine(clear_tags[i][0], query, ban_list[clear_tags[i][0]] + words, slash_words)
                    # word = get_closest_chebyshev(clear_tags[i][0], query, ban_list[clear_tags[i][0]] + words, slash_words)
                    #word = get_closest_braycurtis(clear_tags[i][0], query, ban_list[clear_tags[i][0]] + words, slash_words)
                    word = get_closest_canberra(clear_tags[i][0], query, ban_list[clear_tags[i][0]] + words, slash_words)

                # Si c'est le premier pos_tag de la phrase
                else:
                    # word = get_closest_euclidian(clear_tags[i][0], query, ban_list[clear_tags[i][0]], slash_words)
                    # word = get_closest_cosine(clear_tags[i][0], query, ban_list[clear_tags[i][0]], slash_words)
                    # word = get_closest_chebyshev(clear_tags[i][0], query, ban_list[clear_tags[i][0]], slash_words)
                    # word = get_closest_braycurtis(clear_tags[i][0], query, ban_list[clear_tags[i][0]], slash_words)
                    word = get_closest_canberra(clear_tags[i][0], query, ban_list[clear_tags[i][0]], slash_words)
                
                # On ajoute le mot trouvé à la liste de mots de la phrase
                words.append(word)

                # On ajoute le pos_tag et les mots associés dans les ban_words
                if(clear_tags[i][0] not in ban_words):
                    ban_words[clear_tags[i][0]] = word.split()
                else:
                    ban_words[clear_tags[i][0]] = ban_words[clear_tags[i][0]] + word.split()

            # On remplace les pos_tags dans la phrase par chaque mot trouvé
            for i in range(len(words)):
                line = line.replace("*" + tags[i], words[i], 1)
            print(line)
            break
