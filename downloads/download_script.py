# doownloads images across classes .. serves as a script to download images across classes and store them in a directory structure that can be used for training.
from google_images_download import google_images_download

response = google_images_download.googleimagesdownload()
arguments = {
   # "keywords": "Acne vulgaris face Indian face skin, Active acne breakouts brown Indian face skin, Pimple inflammation South Asian Indian face skin, Post-inflammatory hyperpigmentation face Indian face skin, Dark acne marks brown Indian face skin, Atrophic acne scars cheeks Indian face skin, Melasma hyperpigmentation face Indian face skin, Dark patches neck acanthosis Indian face skin, Uneven skin tone pigmentation South Asian Indian face skin, Tinea corporis ringworm Indian face skin, Fungal infection skin patches brown Indian face skin, Eczema dry scaly patches Indian face skin, Atopic dermatitis flexural brown Indian face skin, Psoriasis plaque violaceous Indian face skin",
    "keywords": "wrinkles face Indian face skin, fine lines face indian face",    
    "limit": 100,
    "print_urls": False
}

paths, errors = response.download(arguments)
print(f"Errors: {errors}")