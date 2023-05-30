from utils import *
project_id = 'som-nero-phi-sywang-starr'
dataset_id = 'imaging'
table_id = 'EncapsulatedDocument_all_3_batches'
fieldnames =['gcc_report_number',
            'signal strength OD','SSOD accuracy',
            'signal strength OS','SSOS accuracy',
            'S-OD','S-OD accuracy',
            'S-OS','S-OS accuracy',
            'SN-OD','SN-OD accuracy',
            'SN-OS','SN-OS accuracy',
            'IN-OD','IN-OD accuracy',
            'IN-OS','IN-OS accuracy',
            'I-OD','I-OD accuracy',
            'I-OS','I-OS accuracy',
            'IT-OD','IT-OD accuracy',
            'IT-OS','IT-OS accuracy',
            'ST-OD','ST-OD accuracy',
            'ST-OS','ST-OS accuracy',
            'Average GCL+IPLThickness OD','Average GCL+IPLThickness OD accuracy',
            'Average GCL+IPLThickness OS','Average GCL+IPLThickness OS accuracy',
            'Minimum GCL+IPLThickness OD','Minimum GCL+IPLThickness OD accuracy',
            'Minimum GCL+IPLThickness OS','Minimum GCL+IPLThickness OS accuracy']

query="""SELECT * FROM `som-nero-phi-sywang-starr.imaging.EncapsulatedDocument_all_3_batches` 
where DocumentTitle like "%OU Ganglion Cell OU Analysis%"
"""
query_job =client.query(query)
df=query_job.to_dataframe()
df.columns = map(str.lower, df.columns)

df_random = df.sample(n=115, random_state=1)
if not os.path.exists('gcc_ou_text_extraction.csv'):
    with open('gcc_ou_text_extraction.csv', 'w', newline ='') as csvfile:
        writer = csv.writer(csvfile, delimiter=',')
        writer.writerow(fieldnames)

        for index in tqdm(range(len(df_random))):    
        # for index in tqdm(range(len(df))):
            dicomfilepath =df_random.iloc[index].dicomfilepath
            # dicomfilepath =df.iloc[index].dicomfilepath
            source_blob_name = dicomfilepath
            output_file_name = 'gcc'+'/' + ('/').join(str(source_blob_name).split('/')[4:])
            if not os.path.exists(output_file_name):
                download_blob('stanfordoptimagroup',source_blob_name, output_file_name)
            ds = pydicom.dcmread(output_file_name,force=True)

            try:

                with open(f'gcc/gcc_ou_{index+1}.pdf', 'wb') as fp:
                    fp.write(ds.EncapsulatedDocument)
                images = convert_from_path(f'gcc/gcc_ou_{index+1}.pdf')

                for i in range(len(images)):
                    # Save pages as images in the pdf
                    images[i].save(f'gcc/gcc_ou_{index+1}.jpg', 'JPEG')

                im = Image.open(f'gcc/gcc_ou_{index+1}.jpg')
                numpydata = np.asarray(im)
                img0 = numpydata[270:315,610:1120]
                img1 = np.concatenate((numpydata[1025:1240,555:810],np.full((20,255,3),255),numpydata[1025:1240,995:1250]), axis=0)
                img2 = np.concatenate((img1,np.full((450,255,3), 255)), axis=1)
                img3 = numpydata[1300:1400,650:1160]
                new_img = np.concatenate((img0,img2,np.full((20,510,3),255),img3), axis=0)

                im = Image.fromarray(np.uint8(new_img))
                im.save(f"gcc/gcc_ou_{index+1}.jpg")

                ocr = PaddleOCR(use_angle_cls=True, lang='en') # need to run only once to download and load model into memory
                img_path = f'gcc/gcc_ou_{index+1}.jpg'
                result = ocr.ocr(img_path, cls=True)
                txts = [line[1][0] for line in result]


                image = Image.open(img_path).convert('RGB')
                boxes = [line[0] for line in result]
                txts = [line[1][0] for line in result]
                scores = [line[1][1] for line in result]
                row = []
                cleaned_text = []
                
                row.append(f'gcc_ou_{index+1}')
                for i, s in enumerate(txts):
                    if ('signal strength') in s.lower():
                        r1 = re.findall(r"\d{1,2}/10",s)
                        if r1 == []:
                            if i == 0:
                                row.extend((txts[i+1],'',txts[i+2],''))
                                cleaned_text.append(('signal strength',txts[i+1],txts[i+2]))
                            if i == 1:
                                row.extend((txts[i+1],'',txts[i-1],''))
                                cleaned_text.append(('signal strength',txts[i+1],txts[i-1]))
                                
                        elif len(r1) == 1:
                            if i == 0:
                                row.extend((r1[0],'', txts[i+1],''))
                                cleaned_text.append(('signal strength',r1[0], txts[i+1]))
                            if i == 1:
                                row.extend((r1[0],'', txts[i-1],''))
                                cleaned_text.append(('signal strength',r1[0], txts[i-1]))
                        else:
                            row.extend((r1[0],'',r1[1],''))
                            cleaned_text.append(('signal strength',r1[0],r1[1]))  
                
                quadrants = ['' for i in range(12)]                
                start_index, end_index = gcc_search_range(txts)
                i = 0
                for j, s in enumerate(txts[start_index:end_index]):
                    if i<12:
                        quadrants[i] = s
                        i += 1                     
                        
                row.extend((quadrants[0],''))
                cleaned_text.append(('S-OD:',quadrants[0]))
                row.extend((quadrants[6],''))
                cleaned_text.append(('S-OS:',quadrants[6]))
                row.extend((quadrants[2],''))
                cleaned_text.append(('SN-OD:',quadrants[2]))
                row.extend((quadrants[7],''))
                cleaned_text.append(('SN-OS:',quadrants[7]))
                row.extend((quadrants[4],''))
                cleaned_text.append(('IN-OD:',quadrants[4]))
                row.extend((quadrants[9],''))
                cleaned_text.append(('IN-OS:',quadrants[9]))
                row.extend((quadrants[5],''))
                cleaned_text.append(('I-OD:',quadrants[5]))
                row.extend((quadrants[11],''))
                cleaned_text.append(('I-OS:',quadrants[11]))
                row.extend((quadrants[3],''))
                cleaned_text.append(('IT-OD:',quadrants[3]))
                row.extend((quadrants[10],''))
                cleaned_text.append(('IT-OS:',quadrants[10]))
                row.extend((quadrants[1],''))
                cleaned_text.append(('ST-OD:',quadrants[1]))
                row.extend((quadrants[8],''))
                cleaned_text.append(('ST-OS:',quadrants[8]))      
                for i, s in enumerate(txts):
                    if ('ave') in s.lower():
                        row.extend((txts[i+1],'',txts[i+2],''))
                        cleaned_text.append(('Average GCL+IPLThickness', txts[i+1], txts[i+2]))
                    if ('min') in s.lower():
                        row.extend((txts[i+1],'',txts[i+2],''))
                        cleaned_text.append(('Minimum GCL+IPLThickness', txts[i+1], txts[i+2]))
                writer.writerow(row)
                cleaned_text.append(ds.DocumentTitle)
                                           
           
                img = Image.open(f'gcc/gcc_ou_{index+1}.jpg')
                width, height = img.size
                myFont = ImageFont.truetype('FreeMono.ttf', 20)
                # Call draw Method to add 2D graphics in an image


                right = 400
                left = 0
                top = 0
                bottom = 400

                new_width = width + right + left
                new_height = height + top + bottom

                result = Image.new(img.mode, (new_width, new_height), (255, 255, 255))

                result.paste(image, (left, top))


                I1 = ImageDraw.Draw(result)
                
                s = ''
                for items in cleaned_text:
                    s = s + '\n' + ' '.join(items)

                # Add Text to an image
                I1.text((new_width - 370, height - 400), s,font=myFont, fill=(255, 0, 0))

                # Save the edited image
                result.save(f"gcc/gcc_ou_result_{index+1}.jpg")  
                
                img_file_name = f'gcc/gcc_ou_{index+1}.jpg'
                result_file_name = f"gcc/gcc_ou_result_{index+1}.jpg"
                
                destination_img_name = 'stanfordoptimagroup/onh_rnfl_sample_100/gcc_ou_img/' + img_file_name.split('/')[1] 
                destination_result_name = 'stanfordoptimagroup/onh_rnfl_sample_100/gcc_ou_result/' + result_file_name.split('/')[1] 
                    
                # upload files to bucket
                os.system(f'gsutil -o GSUtil:parallel_composite_upload_threshold=150M -m cp {img_file_name} gs://{destination_img_name}')
                os.system(f'gsutil -o GSUtil:parallel_composite_upload_threshold=150M -m cp {result_file_name} gs://{destination_result_name}')
                os.system('rm -rf gcc/*')
            except:
                print('-----------')

upload_blob('stanfordoptimagroup', 'gcc_ou_text_extraction.csv', 'gcc_ou_text_extraction.csv',verbose=False)
onh_rnfl_text_extraction = pd.read_csv('gcc_ou_text_extraction.csv')
onh_rnfl_text_extraction = pyarrow_dtype_format_correction(onh_rnfl_text_extraction)
saved_table_id = 'imaging'+f".gcc_ou_text_extraction"
onh_rnfl_text_extraction.to_gbq(saved_table_id, project_id = 'som-nero-phi-sywang-starr', if_exists = 'replace') 
