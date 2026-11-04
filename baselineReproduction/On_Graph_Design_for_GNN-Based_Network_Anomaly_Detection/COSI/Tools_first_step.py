import random
from timeit import default_timer as timer
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import re
import seaborn as sns
import nltk
import plotly.graph_objects as go


"""This function laod the dataset and remove the Unnamed columns at the begenning"""
def load_data(path_dataset):
    df= pd.read_csv(path_dataset)
    columns=df.columns
    str_remove_go="Unnamed"+".*"
    regex_str=re.compile(str_remove_go)
    not_good_col=list(filter(regex_str.match, columns))
    col_keeper=[]
    for i in columns:
        if i not in not_good_col:
            col_keeper.append(i)
        
    new_list=list(col_keeper)

    df=df[new_list]
    print(df.shape)
    return df, columns


"""This function will remove the Unnamed functions that are created sometimes when I recall pd csv"""
def rmv_Unnamed(columns):

    str_remove_go="Unnamed"+".*"
    regex_str=re.compile(str_remove_go)
    not_good_col=list(filter(regex_str.match, columns))
    col_keeper=[]
    for i in columns:
        if i not in not_good_col:
            col_keeper.append(i)
        
    new_list=list(col_keeper)

    return new_list

""" This function will split the feature names into a list of tokens"""
def split_words(col_name):
    words=[]
    for j in re.split('[_ { } ,]',col_name):
        if j not in words:
                words.append(j)
    
    return words

"""This function will split all features into one list of tokens"""
def split_all(features):
    txt=[]
    for i in range(len(features)):
        a=split_words(features[i])
        txt.append(a)
    return txt

"""This function return tokenize list of word from a path of a text 
!! This function is specific for my text, and my documentation, we probably need to modify it for each dataset we want to use,
depending if numbers, {}, = are important or not for example"""
def cleaning_txt_documentation(path):

    with open(path, "r", encoding="utf-8") as file:
        text = file.read()


    sentences = nltk.sent_tokenize(text)
    df_sentences=pd.DataFrame(sentences)
    cleaned_txt=[]
    for w in range(len(df_sentences)):
        clean=df_sentences.iloc[w,0].lower()
        clean=re.sub("[_ { } , #] "," ", clean)
        clean=re.sub(r'\n|\.|\_|\{|\}', ' ', clean)
        cleaned_txt.append(clean)

    df_sentences[0]=cleaned_txt
    new_dico_all_word=[]
    for raws in df_sentences[0]:
        corpus=raws.split(" ")
        new_dico_all_word.append(corpus)
    return new_dico_all_word


"""This function will find the frequence of each words and give a dictionary with key each words, frequance as value"""
def find_frequence_of_words(list_col):
    words_freq={}
    words_dico=[]
    for k in range(len(list_col)):
        for j in re.split('[_ { } ,]',list_col[k]):
            if j not in words_freq:
                words_freq[j]=1
                words_dico.append(j)
            else:
                 words_freq[j]+=1
    
    return words_freq, words_dico

"""estimate the total number of term in all features"""
def estimate_total_number_of_word(features):
    num=0
    for name, number in find_frequence_of_words(features)[0].items():
        num+=number
    return num


"""This function is supposed to estimate the number of document in which each words appear, but it is equal to the number total of appearing so not useful"""
def number_a_word_in_doc(features, freq_each, dico_word):
    dico_for_number_of_time_each_word_appear_in_doc={}
    for name, freq in freq_each.items():
        for k in range(len(features)):
            if name in split_words(features[k]):
                print(f'name : {name}')
                print(f'sequence: {split_words(features[k])}')
                if name not in dico_for_number_of_time_each_word_appear_in_doc:
                    dico_for_number_of_time_each_word_appear_in_doc[name]=1
                else:
                    dico_for_number_of_time_each_word_appear_in_doc[name]+=1
    
    return dico_for_number_of_time_each_word_appear_in_doc


""" Average of word2vec vectors with kind of TF-IDF score to go from words to sentence (features names here)"""
def TF_IDF(word, freq_each, number_total):
    score=0
    number_of_this_word=freq_each[word]
    #the log seams way better
    idf=np.log(number_total/number_of_this_word)
    #iddf=number_of_this_word/number_total
    score=idf

    return score
""" Made my own score for agregation of each token in my sentence : current version used """
def TF_new_version(word, freq_each, number_total):
    score=0
    number_of_this_word=freq_each[word]
    #the log seams way better
    idf2=(number_total/number_of_this_word)*100
    #iddf=number_of_this_word/number_total
    score=idf2

    return score

"""Made my own cosine similarity score"""
def cosine_similarity(seq1, seq2):
    # Convert sequences to numpy arrays
    vec1 = np.array(seq1)
    vec2 = np.array(seq2)
    
    # Compute dot product
    dot_product = np.dot(vec1, vec2)
    
    # Compute magnitudes
    mag1 = np.linalg.norm(vec1)
    mag2 = np.linalg.norm(vec2)
    
    # Compute cosine similarity
    similarity = dot_product / (mag1 * mag2)
    
    return similarity

#deprecated : output not in a good shape for plot
def score_similarity(columns, model_word2vec, verbose=0):

    dico=find_frequence_of_words(columns)[0]
    total_number=estimate_total_number_of_word(columns)

    mat_sim=[]
    mat_min_sim=[]
    mat_nlog=[]
    for coli in range(len(columns)):
        if verbose==1:
            print(f'for feature : {columns[coli]}')
        mat_sim_col=[]
        mat_mi_sim_col=[]
        mat_nlog_sim_col=[]
        for colj in range(len(columns)):
            if coli==colj:
                continue
            if verbose==1:
                print(f'for feature : {columns[colj]}')
            words_i=split_words(columns[coli])
            words_j=split_words(columns[colj])

            score_i=0
            score_i_avg=0
            score_i_3=0
            
            for word1 in words_i:
                 idf1=TF_IDF(word1,dico,total_number)
                 idfi=TF_new_version(word1,dico,total_number)
                 score_i+=model_word2vec.wv[word1]*idf1
                 score_i_3+=model_word2vec.wv[word1]*idfi
                 score_i_avg+=model_word2vec.wv[word1]

            score_i=score_i/len(words_i)
            score_i_avg=score_i_avg/len(words_i)
            score_i_3=score_i_3/len(words_i)

            score_j=0
            score_j_avg=0
            score_j_3=0

            for word2 in words_j:   
                idf2=TF_IDF(word2,dico,total_number)
                idfj=TF_new_version(word2,dico,total_number)
                score_j+=model_word2vec.wv[word2]*idf2
                score_j_3+=model_word2vec.wv[word2]*idfj
                score_j_avg+=model_word2vec.wv[word2]

            score_j=score_j/len(words_j)
            score_j_3=score_j_3/len(words_j)
            score_j_avg=score_j_avg/len(words_j)

            similarity_score=cosine_similarity(score_i, score_j)
            sim_avg=cosine_similarity(score_i_avg, score_j_avg)
            similarity_nlog=cosine_similarity(score_i_3, score_j_3)


            mat_mi_sim_col.append(sim_avg)                
            mat_sim_col.append(similarity_score)
            mat_nlog_sim_col.append(similarity_nlog)

        mat_min_sim.append(mat_mi_sim_col)
        mat_sim.append(mat_sim_col)
        mat_nlog.append(mat_nlog_sim_col)
    return mat_sim, mat_min_sim, mat_nlog
                    
#I will modify a bit the function to have only what is necessary (mat3 and *in the shape needed below) (dont remember what this comment means)
""" return 2 dictionnary : one with similarity score of each features names, the other with there own score : this is my node placement in the graph"""
def score_similarity_current(columns, model_word2vec, verbose=0):

    dico=find_frequence_of_words(columns)[0]
    total_number=estimate_total_number_of_word(columns)

    mat_nlog={}
    vector_names={}
    for coli in range(len(columns)):
        if verbose==1:
            print(f'for feature : {columns[coli]}')

        for colj in range(len(columns)):
            if coli==colj:
                continue
            if verbose==1:
                print(f'for feature : {columns[colj]}')
            words_i=split_words(columns[coli])
            words_j=split_words(columns[colj])

            score_i_3=0
            
            for word1 in words_i:
                 idfi=TF_new_version(word1,dico,total_number)
                 score_i_3+=model_word2vec.wv[word1]*idfi

            score_i_3=score_i_3/len(words_i)

            score_j_3=0

            for word2 in words_j:   
                idfj=TF_new_version(word2,dico,total_number)
                score_j_3+=model_word2vec.wv[word2]*idfj

            score_j_3=score_j_3/len(words_j)

            vector_names[columns[colj]]=score_j_3

            similarity_nlog=cosine_similarity(score_i_3, score_j_3)

            mat_nlog[(columns[coli], columns[colj])]= similarity_nlog

    return mat_nlog, vector_names
                    

#deprecated : not using anymore list but rather a dictonary
def get_x_axis(vectore_results, x_axis):
    x=[]
    for colname, placement  in vectore_results.items():
        x.append(placement[x_axis])

    return x


#current version that I use
""" Take the correspondant axis and the output is the list of the axis of each nodes """
def get_x_axis_current(vectore_results, x_axis):
    x={}
    for colname, placement  in vectore_results.items():
        x[colname]=placement[x_axis]

    return x



""" This function get x,y and z axis for all edges in the graph, print the total amount off edges in the graph and all nodes sort by names entry """
def get_edges_v2(cosin_score, threashold, vectore_results):
    x,y,z=[],[],[]
    total_counter_edges=0

    x_frst=list(vectore_results.values())[0][0]
    #print(x_frst)
    y_frst=list(vectore_results.values())[0][1]
    #print(y_frst)
    z_frst=list(vectore_results.values())[0][2]
    #print(z_frst)

    all_nodes=[]
    X_per_node=[]
    for (name1, name2),score in cosin_score.items():
        
        x_memory=x_frst
        y_memory=y_frst
        z_memory=z_frst

        if score>threashold:

                total_counter_edges+=1
                x.append(vectore_results[name1][0])
                y.append(vectore_results[name1][1])
                z.append(vectore_results[name1][2])

                x.append(vectore_results[name2][0])
                y.append(vectore_results[name2][1])
                z.append(vectore_results[name2][2])


                if (x[-2]==x_memory) and (y[-2]==y_memory) and (z[-2]==z_memory):
                    X_per_node.append((name1,name2))
                else:
                    x_frst, y_frst, z_frst=x[-2], y[-2], z[-2]
                    all_nodes.append(X_per_node)
                    X_per_node=[]
        

    print(f' total amount of edges : {total_counter_edges}')
    return x,y,z, all_nodes


#The only effect of this function comparing to v2 is that I can add on my nodes only a part of the data that correspond to specific word in the name of the features
def get_edges_v3(cosin_score, threashold, vectore_results, regex_word, both_match=True):
    x,y,z=[],[],[]
    total_counter_edges=0

    x_frst=list(vectore_results.values())[0][0]
    y_frst=list(vectore_results.values())[0][1]
    z_frst=list(vectore_results.values())[0][2]
    r_express=".*"+regex_word+".*"
    all_nodes=[]
    X_per_node=[]
    for (name1, name2),score in cosin_score.items():
        if re.match(r_express, name1):
            if both_match:
                if re.match(r_express, name2):
            
                    x_memory=x_frst
                    y_memory=y_frst
                    z_memory=z_frst

                    if score>threashold:

                            total_counter_edges+=1
                            x.append(vectore_results[name1][0])
                            y.append(vectore_results[name1][1])
                            z.append(vectore_results[name1][2])

                            x.append(vectore_results[name2][0])
                            y.append(vectore_results[name2][1])
                            z.append(vectore_results[name2][2])


                            if (x[-2]==x_memory) and (y[-2]==y_memory) and (z[-2]==z_memory):
                                X_per_node.append((name1,name2))
                            else:
                                x_frst, y_frst, z_frst=x[-2], y[-2], z[-2]
                                all_nodes.append(X_per_node)
                                X_per_node=[]
                else:
                    continue

            else:

                            
                    x_memory=x_frst
                    y_memory=y_frst
                    z_memory=z_frst

                    if score>threashold:
                            
                            total_counter_edges+=1
                            x.append(vectore_results[name1][0])
                            y.append(vectore_results[name1][1])
                            z.append(vectore_results[name1][2])

                            x.append(vectore_results[name2][0])
                            y.append(vectore_results[name2][1])
                            z.append(vectore_results[name2][2])


                            if (x[-2]==x_memory) and (y[-2]==y_memory) and (z[-2]==z_memory):
                                X_per_node.append((name1,name2))
                            else:
                                x_frst, y_frst, z_frst=x[-2], y[-2], z[-2]
                                all_nodes.append(X_per_node)
                                X_per_node=[]
        else:
            continue

        

    print(f' total amount of edges : {total_counter_edges}')
    return x,y,z, all_nodes



"""This function is only used for the topK function below, there is some values that are not counted correctly by this one, and we need the function adding to avoid
error by adding the feature name missing in the dictionnary"""
def adding(name, list_scored, cosin_score, both_match, r_express, threashold):

    last_process=None
    for (name1, name2),score in cosin_score.items():
        if name1==name:
            if re.match(r_express, name):
                if both_match:
                    if re.match(r_express, name2):
                        if score>threashold:
                            last_process=name2

        
    total_counter_edges=0
    new_list= []
    for (name1, name2),score in cosin_score.items():
        if name1==name:
            if re.match(r_express, name):
                if both_match:
                    if re.match(r_express, name2):
                        if score>threashold:
                            if name2 != last_process:
                                total_counter_edges+=1
                                new_list.append(score)
                            else:
                                total_counter_edges+=1
                                new_list.append(score)
                                list_scored[name]=new_list
    return list_scored


"""This function is a get_edges with top(K), to select the topK score aound each neighbours"""
def get_edges_topK(cosin_score, threashold, vectore_results, regex_word,topK=200, both_match=True):
    x,y,z=[],[],[]
    total_counter_edges=0

    r_express=".*"+regex_word+".*"
    all_nodes=[]
 
    first_name=list(cosin_score.keys())[0][0] #in fact it is 'previous name' and not the first
    last_name=list(cosin_score.keys())[-1][0]


    scored_of_all={}
    new_list=[]


    #need to get the last qualified pair for the next part
    last_qualified_pair = None

    for (name1, name2),score in cosin_score.items():
        if re.match(r_express, name1):
            if both_match:
                if re.match(r_express, name2):
                    if score>threashold:
                        last_qualified_pair = (name1, name2)



#get the score for each nodes specifically
    for (name1, name2),score in cosin_score.items():
        if re.match(r_express, name1):
            if both_match:
                if re.match(r_express, name2):
                    if score>threashold:
 
                        if name1==first_name:
                            total_counter_edges+=1
                            new_list.append(score)

                        else:
                            if name1==last_name:
                                if (name1,name2)==last_qualified_pair :
                                    total_counter_edges+=1
                                    new_list.append(score)
                                    scored_of_all[name1]=new_list
                                else:
                                    total_counter_edges+=1
                                    new_list.append(score)

                            

                            else:
                                total_counter_edges=0
                                scored_of_all[first_name]=new_list

                                new_list=[]
                                first_name=name1
                                total_counter_edges+=1
                                new_list.append(score)






#make a topK
    for name, new_score_list in scored_of_all.items():
        #print(len(new_score_list))
        if len(new_score_list)>topK:
            new_score_list=sorted(new_score_list, reverse=True)[:topK]
            scored_of_all[name]=new_score_list
            #print(new_score_list)


#modify the score function to give 0 instead of the previous values for those that are not selected by topK
    for (name1, name2),score in cosin_score.items():
        if re.match(r_express, name1):
            if both_match:
                if re.match(r_express, name2):
                    if score>threashold:
                        #print(len(scored_of_all[name1]))
                        if name1 not in scored_of_all:
                            scored_of_all=adding(name1,scored_of_all, cosin_score, both_match, r_express, threashold)
                           
                        if score not in scored_of_all[name1]:
                            cosin_score[(name1,name2)]=0



#call the classic function get edges with new scores
    x,y,z,all_nodes=get_edges_v3(cosin_score, threashold, vectore_results, regex_word, both_match)
    return x,y,z, all_nodes




"""give the mean of the number of edges per nodes"""
def get_medium_number_of_edges(total_edges):
    med=0
    for k in range(len(total_edges)):
        med+=len(total_edges[k])

    med=med/len(total_edges)
    return med


"""plot the distribution of the number of edges per nodes"""
def get_distribution_number_of_edges_per_nodes(word, cosine_score, thd, vectors, ploting=True):
    xe,ye,ze, total_edges=get_edges_v3(cosine_score,thd, vectors, word)
    size_nodes=len(get_node_axis(word,vectors,0))
    nb=[]
    for k in range(len(total_edges)):
        nb.append(len(total_edges[k]))

    size_nb=len(nb)
    diff=size_nodes-size_nb
    if diff>0:
        for i in range(diff):
            nb.append(0)

    if ploting:
        nb=sorted(nb)
        hist,bins=np.histogram(nb, bins=np.linspace(0, max(nb),int(size_nb/8))) #### here I put 8 but we should change it sometimes
        sns.histplot(nb,bins=bins ,kde=False, color='blue', edgecolor='black')

    return nb


"""modification of get_x_axis that can work on a subspace with regex"""
def get_node_axis(word, vectore_results, axis):
    x={}
    regex_expression=".*"+word+".*"
    for colname, placement  in vectore_results.items():
        if re.match(regex_expression, colname):
            x[colname]=placement[axis]

    return x


"""to plot the whole dataset graph"""
def plot_graph_v1(cosine_score,threashold, vectors):
    x1=get_x_axis_current(vectors, 0)
    y1=get_x_axis_current(vectors, 1)
    z1=get_x_axis_current(vectors, 2)

    xe,ye,ze, useless=get_edges_v2(cosine_score,threashold, vectors)
    # Create Plotly figure
    fig = go.Figure()

    # Add nodes as scatter points
    node_trace = go.Scatter3d(
        x=list(x1.values()),  # x-coordinates of nodes
        y=list(y1.values()),  # y-coordinates of nodes
        z=list(z1.values()),  # z-coordinates of nodes (for 3D)
        mode='markers',
        text=list(x1.keys()),
        marker=dict(symbol='circle', size=2, color='blue')
    )
    fig.add_trace(node_trace)

    # Add edges as lines
    edge_trace = go.Scatter3d(
        x=xe,  # x-coordinates of edges
        y=ye,  # y-coordinates of edges
        z=ze,  # z-coordinates of edges (for 3D)
        mode='lines',
        line=dict(width=1, color='black'),
        hoverinfo='none'
    )
    fig.add_trace(edge_trace)

    # Customize layout
    fig.update_layout(
        title='3D Network Graph',
        scene=dict(
            xaxis=dict(title='X'),
            yaxis=dict(title='Y'),
            zaxis=dict(title='Z'),
        )
    )

    # Show plot
    fig.show()
    return


""" plot a part of the dataset that correspond to the word in feature name """
def plot_only_part_data(cosine_score,threashold, vectors, word=""):

    x1=get_node_axis(word,vectors, 0)
    y1=get_node_axis(word,vectors, 1)
    z1=get_node_axis(word,vectors, 2)

    xe,ye,ze,all_edges=get_edges_v3(cosin_score=cosine_score,threashold=threashold, vectore_results=vectors, regex_word=word)
    

    # Create Plotly figure
    fig = go.Figure()

    # Add nodes as scatter points
    node_trace = go.Scatter3d(
        x=list(x1.values()),  # x-coordinates of nodes
        y=list(y1.values()),  # y-coordinates of nodes
        z=list(z1.values()),  # z-coordinates of nodes (for 3D)
        mode='markers',
        text=list(x1.keys()),
        marker=dict(symbol='circle', size=2, color='blue')
    )
    fig.add_trace(node_trace)

# Add edges as lines
    edge_trace = go.Scatter3d(
        x=xe,  # x-coordinates of edges
        y=ye,  # y-coordinates of edges
        z=ze,  # z-coordinates of edges (for 3D)
        mode='lines',
        line=dict(width=1, color='black'),
        hoverinfo='none'
    )
    fig.add_trace(edge_trace)

    # Customize layout
    fig.update_layout(
        title='3D Network Graph',
        scene=dict(
            xaxis=dict(title='X'),
            yaxis=dict(title='Y'),
            zaxis=dict(title='Z'),
        )
    )

    # Show plot
    fig.show()
    return


"""Print the heatmap of the correlation matrix, not great unfortunately : to do better"""
def print_corr_mat(mtx):
  matrix = mtx

  #plotting correlation matrix 
  plt.imshow(matrix, cmap='Blues')

  #adding colorbar 
  plt.colorbar()

  #extracting variable names 
  variables = []
  for i in matrix.columns:
    variables.append(i)

  # Adding labels to the matrix
  plt.xticks(range(len(matrix)), variables, rotation=45, ha='right')
  plt.yticks(range(len(matrix)), variables)

  # Display the plot
  plt.show()
  return

"""change my correlation matrix into a dictionnary"""
def change_into_dico(mat):
    a = mat.values
    a[np.tril_indices(a.shape[0], 0)] = np.nan
    mat[:] = a 
    x=mat.stack()
    out = dict(x.items())
    return out


##warning : the similarity dictionnary have both directions keys, but not the correlation dictionnary
"""give a new score of proximity between two features, based on name proximity : sim_score, and correlation : corr """
def build_new_score_product(corr, sim):
    new_dictionnary={}
    for (name1, name2), score in corr.items():
        new_dictionnary[(name1,name2)]=abs(sim[(name1, name2)]*score)
    
    return new_dictionnary

"""give a new score with a lambda mixing parameter between similarity and correlation """
def build_score_lambda(corr,sim,L):
    new_dictionnary={}
    for (name1, name2), score in corr.items():
        new_dictionnary[(name1,name2)]=L*abs(sim[(name1, name2)])+(1-L)*abs(score)
    
    return new_dictionnary

def edges_dico(final_dico, threashold):
    a={}
    for (name1, name2), score in final_dico.items():
        if score>=threashold:
            a[(name1,name2)]= score

    return a



#current version
""" This function build the adjacency matrix that we are looking for"""
def build_adjacency_matrix_v2(edges_final, mat_corr):
    Adjacency_matrix=[]
    names=mat_corr.columns.values
    end_egdes=len(edges_final)
    for k in range(len(mat_corr)):
        add_line=[]
        name1=names[k]
        for j in range(len(mat_corr.iloc[0])):
            name2=names[j]
            x=(name1, name2)
            i=0
            while i<end_egdes:
                if (x) in edges_final[i]:
                    add_line.append(1)
                    i=end_egdes+1
                else:
                    i+=1
            if i==end_egdes:
                add_line.append(0)

        #print(f'j : {j}, size line : {len(add_line)}')
        Adjacency_matrix.append(add_line)
        

    #update : when j=k =>1
    for i in range(len(mat_corr)):
        Adjacency_matrix[i][i]=1
        

    #now I need a symetric matrix:
    for k in range(len(Adjacency_matrix)):
        for j in range(len(Adjacency_matrix[0])):
            if Adjacency_matrix[k][j]==1:
                Adjacency_matrix[j][k]=1
                
    return Adjacency_matrix


"""This function will estimate the difference between two adjacency matrix"""
def divergence_matrix(A,B):
    score=0
    if A.shape !=B.shape:
        return "A and B must be the same size" 
    else:
        for k in range(len(A)):
            for j in range(len(A[0])):
                if A[k][j]!=B[k][j]:
                    score+=1
        score =score/(len(A)**2)
        return score

