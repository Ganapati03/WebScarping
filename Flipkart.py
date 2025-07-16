import requests
from bs4 import BeautifulSoup
import pandas as pd

headers = {
    "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
                  "AppleWebKit/537.36 (KHTML, like Gecko) "
                  "Chrome/91.0.4472.124 Safari/537.36"
}

Product_name = []
SmartPhone_Prices = []
Descriptions = []
Reviews = []


for i in range(2, 23):
    url = "https://www.flipkart.com/search?q=smart+phone+under+50000&as=on&as-show=on&otracker=AS_Query_OrganicAutoSuggest_1_24_na_na_na&otracker1=AS_Query_OrganicAutoSuggest_1_24_na_na_na&as-pos=1&as-type=RECENT&suggestionId=smart+phone+under+50000&requestId=456a6ab2-c265-4070-8284-00a7bf872d22&as-searchtext=smart+phone+under+50000&page=" + str(i)

 
    r = requests.get(url, headers=headers)
    print("Status Code:", r.status_code)

  
    soup = BeautifulSoup(r.text, "lxml")

    box = soup.find("div", class_="DOjaWF gdgoEp")
    if not box:
        print(f"No product container found on page {i}")
        continue

  
    names = box.find_all("div", class_="KzDlHZ")
    for name_tag in names:
        name = name_tag.text.strip()
        Product_name.append(name)


    for idx, name in enumerate(Product_name, start=1):
        print(f"{idx}. Product Name: {name}")

   
    prices = box.find_all("div", class_="Nx9bqj _4b5DiR")
    for price_tag in prices:
        price = price_tag.text.strip()
        SmartPhone_Prices.append(price)

   
    for idx, price in enumerate(SmartPhone_Prices, start=1):
        print(f"{idx}. Price: {price}")

    
    descs = box.find_all("ul", class_="G4BRas")
    for desc_tag in descs:
        desc = desc_tag.text.strip()
        Descriptions.append(desc)

  
    for idx, specs in enumerate(Descriptions):
        print(f"\n{idx}. Specifications: {specs}")

   
    ratings = box.find_all("div", class_="XQDdHH")
    for rate_tag in ratings:
        rate = rate_tag.text.strip()
        Reviews.append(rate)

  
    for idx, rate in enumerate(Reviews):
        print(f"{idx}. Ratings: {rate}")


max_len = max(len(Product_name), len(SmartPhone_Prices), len(Descriptions), len(Reviews))

def pad_list(lst):
    return lst + ["NA"] * (max_len - len(lst))

Product_names = pad_list(Product_name)
Smart_phone_price = pad_list(SmartPhone_Prices)
Des = pad_list(Descriptions)
Rev = pad_list(Reviews)


df = pd.DataFrame({
    "Product Name": Product_names,
    "SmartPhone Price": Smart_phone_price,
    "Descriptions": Des,
    "Reviews": Rev
})

print(df)


df.to_csv("C:/Users/ganap/OneDrive/Desktop/Scraped Data/DatamobileAnalysis.csv", index=False)