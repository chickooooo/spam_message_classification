import scripts.utils as utils

# setup the model and other dependencies
setup = utils.Setup()

# data to test on
test_data = [
    "URGENT! You have won a 1 week FREE membership in our £100,000 Prize Jackpot! Txt the word: CLAIM to No: 81010 T&C www.dbuk.net LCCLTD POBOX 4403LDNW1A7RW18",
    "Nah I don't think he goes to usf, he lives around here though"
]

# make predictions
result = setup.predict(test_data)
# print result
print(result)
