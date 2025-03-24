import os
import shutil
import glob

n = 194

# ce script permet de changer le numero des classes et de leur attribuer 0 à tous les arbres la classe 0
# for i in range(1,n):
#     file_txt = f"C:/Users/rahim/Deeplearning_oct_2024/Pleiade_2023_geo/Pleiade_Vue1_texte/image{i}.txt"

#     output_file = f"C:/Users/rahim/Deeplearning_oct_2024/Pleiade_2023_geo/Pleiade_Vue1_sub/image{i}.txt"  # Optional: Output file path

#     # Open the input file and create an output file
#     with open(file_txt, "r") as infile, open(output_file, "w") as outfile:
#         for line in infile:
#             # Split the line into parts (assuming space-separated numbers)
#             parts = line.split()
#             if parts:  # Ensure the line is not empty
#                 parts[0] = "0"  # Change the first number to 0
#             # Write the modified line to the output file
# #             outfile.write(" ".join(parts) + "\n")
# #     print(f"Modified file saved to {output_file}")
# number = 1



# for i in range(n,400):

#     file_txt = f"C:/Users/rahim/Deeplearning_oct_2024/Pleiade_2023_geo/Pleiade_Vue3_2023_traitement/Pleiade_Vue3_texte/image{i}.txt"

#     output_file = f"C:/Users/rahim/Deeplearning_oct_2024/Pleiade_2023_geo/Pleiade_Vue3_2023_traitement/Pleiade_Vue3_classe0/image{i}.txt" 
#     with open(file_txt, "r") as infile:
#         lines = infile.readlines()
            
#     lines_to_keep = [line for line in lines if not line.startswith('1')]
#     if len(lines_to_keep) != 0:
#         with open(output_file, "w") as outfile:
#             outfile.writelines(lines_to_keep)
#             print(f"Modified file saved to {output_file}")




'''

Maintenant on va associer chaque fichier texte avec son image correspondante

'''

j = 181

for i in range(n,400):

    file_img = f"C:/Users/rahim/Deeplearning_oct_2024/Pleiade_2023_geo/Pleiade_Vue3_2023_traitement/Pleiade_Vue3_image_png/image{i}.png"
    output_file_img = f"C:/Users/rahim/Deeplearning_oct_2024/Pleiade_2023_geo/Pleiade_Vue3_2023_traitement/Pleiade_Vue3_final_img/image{j}.png"
    output_file_txt = f"C:/Users/rahim/Deeplearning_oct_2024/Pleiade_2023_geo/Pleiade_Vue3_2023_traitement/Pleiade_Vue3_final_txt/image{j}.txt"
    file_txt = f"C:/Users/rahim/Deeplearning_oct_2024/Pleiade_2023_geo/Pleiade_Vue3_2023_traitement/Pleiade_Vue3_classe0/image{i}.txt" 

    
    if os.path.exists(file_txt):
        print(file_txt)
        shutil.copy(file_txt, output_file_txt)
        shutil.copy(file_img, output_file_img)
        j+=1


    else:

        print("File not found!",file_txt)

        