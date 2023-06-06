#Define as váriaveis globais


#Caminho para o executável do OCR
tesseractOCR_path = "C:\Program Files\Tesseract-OCR\Tesseract.exe"

#Caminho para o poppler - lib para leitura de arquivos pdf
poppler_path = "C:\Program Files\poppler-0.68.0\bin"

#Nome do modelo de detecção
detection_model_name = 'structures_FRCNN'


#Percentual mínimo que as bolas devem estar marcadas para serem consideradas corretas
minpercent = 0.8


#Variáveis relacionadas ao mapeamento da grade de respostas

#Dimensões da página do cartão resposta
answer_card_shape = (1653,2339)


#Dimensões da grade de resposta
grade_shape = (1505,395)


#Area da grade - 1505 * 395 = 594.475   - Para garantir que a grade será detectada utiliza-se um valor menor.
grade_area = 500e3


#Posições arbitrárias das bolhas de marcação com relação a grade de acordo com o template utilizado

#Posição arbitrária do 1o centro 
first_center_x, first_center_y = (62,57)

#Espaços entre as bolhas eixo x: 36
x_bubble_space = 36

#Espaços entre as bolhas eixo y: 32
y_bubble_space = 32

#Espaços entre as regiões da grade : 184
region_space = 184

#Diametro da bolha
bubble_diameter = 22

#Número de colunas de cada região da grade
n_grade_cols = 4

#Número de linhas da grade
n_grade_lines = 10

#Número de regiões
n_grade_regions = 8