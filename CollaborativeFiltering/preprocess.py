import numpy as np
import pandas as pd

def load_train_set():
    users_info = pd.read_csv("./users.csv")
    business_info = pd.read_csv("business.csv")
    review_info = pd.read_csv("train_reviews.csv")
    
    business_ids = pd.DataFrame.as_matrix(business_info["business_id"])
    user_ids = pd.DataFrame.as_matrix(users_info["user_id"])
    #create a dataframe to store a user business mapping
    rate_matrix = pd.DataFrame(columns = business_ids, index = user_ids,dtype=float)
    rate_matrix = rate_matrix.fillna(0)
    R = np.zeros((user_ids.shape[0], business_ids.shape[0]))

    for idx, row in review_info.iterrows():
        rate_matrix.at[row["user_id"],row["business_id"]] = row["stars"]
        r_idx = np.where(user_ids == row["user_id"])[0][0]
        c_idx = np.where(business_ids == row["business_id"])[0][0]
        R[r_idx][c_idx] = 1
    print(rate_matrix[0:3])
    #create a hash table for {user : {business_id : rate}}
    # user_hash = {}
    # bus_hash = {}
    # for _,row in review_info.iterrows():
    #     user_id = row[row["user_id"]]
    #     business_id = row["business_id"]
    #     if  user_id in user_hash:
    #         user_hash[user_id][business_id] = row["star"]
    #     else:
    #         user_hash[user_id] = {business_id : row["star"]}
        
    #     if business_id in bus_hash:
    #         bus_hash[business_id][user_id] = row["star"]
    #     else:
    #         bus_hash[business_id] = {user_id : row["star"]}

    return user_ids, business_ids, rate_matrix, R

def load_valid_set():
    valid_set = pd.DataFrame.as_matrix(pd.read_csv("./validate_queries.csv"))
    user_ids = valid_set[:,1]
    bus_ids = valid_set[:,2]
    y_t = valid_set[:,3]
 

    return user_ids, bus_ids, y_t

def main():
    load_valid_set()

if __name__ == "__main__":
    main()
