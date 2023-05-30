from utils import *
project_id = 'som-nero-phi-sywang-starr'
dataset_id = 'imaging'
table_id = 'EncapsulatedDocument_all_3_batches'
fieldnames =['onh_rnfl_report_number',
            'signal strength OD','SSOD accuracy',
            'signal strength OS','SSOS accuracy',
            'average rnfl thickness OD','AvgRNFLThickOD accuracy',
            'average rnfl thickness OS','AvgRNFLThickOS accuracy',
            'RNFL Symmetry','RNFL Symmetry accuracy',
            'Rim Area OD','RAOD accuracy',
            'Rim Area OS','RAOS accuracy',
            'Disc Area OD','DAOD accuracy',
            'Disc Area OS','DAOS accuracy',
            'Average C/D Ratio OD','Avg C/D ratio OD accuracy',
            'Average C/D Ratio OS', 'Avg C/D ratio OS accuracy',
            'Vertical C/D Ratio OD','Vert C/D ratio OD accuracy',
            'Vertical C/D Ratio OS','Vert C/D ratio OS accuracy',
            'Cup Volume OD','CVOD accuracy',
            'Cup Volume OS','CVOS accuracy',
            'S-OD','S-OD accuracy',
            'S-OS','S-OS accuracy',
            'T-OD','T-OD accuracy',
            'T-OS','T-OS accuracy',
            'N-OD','N-OD accuracy',
            'N-OS','N-OS accuracy',
            'I-OD','I-OD accuracy',
            'I-OS','I-OS accuracy',
            '1-OD','1-OD accuracy',
            '1-OS','1-OS accuracy',
            '2-OD','2-OD accuracy',
            '2-OS','2-OS accuracy',
            '3-OD','3-OD accuracy',
            '3-OS','3-OS accuracy',
            '4-OD','4-OD accuracy',
            '4-OS','4-OS accuracy',
            '5-OD','5-OD accuracy',
            '5-OS','5-OS accuracy',
            '6-OD','6-OD accuracy',
            '6-OS','6-OS accuracy',
            '7-OD','7-OD accuracy',
            '7-OS','7-OS accuracy',
            '8-OD','8-OD accuracy',
            '8-OS','8-OS accuracy',
            '9-OD','9-OD accuracy',
            '9-OS','9-OS accuracy',
            '10-OD','10-OD accuracy',
            '10-OS','10-OS accuracy',
            '11-OD','11-OD accuracy',
            '11-OS','11-OS accuracy',
            '12-OD','12-OD accuracy',
            '12-OS','12-OS accuracy']

query="""SELECT *
FROM `{project_id}.{dataset_id}.{table_id}`where DocumentTitle like '%Cirrus_OU_ONH and RNFL OU Analysis%'
 """.format_map({'project_id': project_id,
                'dataset_id': dataset_id,
                'table_id': table_id})
query_job =client.query(query)
df=query_job.to_dataframe()
df.columns = map(str.lower, df.columns)

#df_random = df.sample(n=158, random_state=3)
if not os.path.exists('onh_rnfl_text_extraction.csv'):
    with open('onh_rnfl_text_extraction.csv', 'w', newline ='') as csvfile:
        writer = csv.writer(csvfile, delimiter=',')
        writer.writerow(fieldnames)

        # for index in tqdm(range(len(df_random))):    
        for index in tqdm(range(len(df))):
            # dicomfilepath =df_random.iloc[index].dicomfilepath
            dicomfilepath =df.iloc[index].dicomfilepath
            source_blob_name = dicomfilepath
            output_file_name = 'onh_rnfl'+'/' + ('/').join(str(source_blob_name).split('/')[4:])
            if not os.path.exists(output_file_name):
                download_blob('stanfordoptimagroup',source_blob_name, output_file_name)
            ds = pydicom.dcmread(output_file_name,force=True)

            try:

                with open(f'onh_rnfl/onh_rnfl_{index+1}.pdf', 'wb') as fp:
                    fp.write(ds.EncapsulatedDocument)
                images = convert_from_path(f'onh_rnfl/onh_rnfl_{index+1}.pdf')

                for i in range(len(images)):
                    # Save pages as images in the pdf
                    images[i].save(f'onh_rnfl/onh_rnfl_{index+1}.jpg', 'JPEG')

                im = Image.open(f'onh_rnfl/onh_rnfl_{index+1}.jpg')
                numpydata = np.asarray(im)
                img0 = numpydata[270:310,600:1221]
                img1 = numpydata[400:790,600:1221]
                img2 = numpydata[1420:1920,600:1221]
                new_img = np.concatenate((img0,img1, img2), axis=0)
                im = Image.fromarray(new_img)
                im.save(f"onh_rnfl/onh_rnfl_{index+1}.jpg")

                ocr = PaddleOCR(use_angle_cls=True, lang='en') # need to run only once to download and load model into memory
                img_path = f'onh_rnfl/onh_rnfl_{index+1}.jpg'
                result = ocr.ocr(img_path, cls=True)
                txts = [line[1][0] for line in result]


                image = Image.open(img_path).convert('RGB')
                boxes = [line[0] for line in result]
                txts = [line[1][0] for line in result]
                scores = [line[1][1] for line in result]
                row = []
                cleaned_text = []
                
                row.append(f'onh_rnfl_{index+1}')
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
                for i, s in enumerate(txts):
                    if ('average') in s.lower() and ('rnfl') in s.lower()  and ('thickness') in s.lower():
                        ct = 0

                        l = []
                        for j, s in enumerate(txts[i:i+4]):
                            if 'mm' in s.lower():
                                continue
                            if 'm' in s.lower() and len(s)<=5:
                                ct+=1

                                tmp_s = ''
                                for c in s:
                                    if c == 'm':

                                        tmp_s+=('\u03BC'+ c)
                                    else:

                                        tmp_s+=c
                                l.extend([tmp_s, ''])
  
                        row.extend(l)
                        cleaned_text.append(('average rnfl thickness', l[0]))
                        cleaned_text.append(('average rnfl thickness', l[2]))
                        if ct == 1:
                            row.extend(['',''])
                            
                for i, s in enumerate(txts):
                    if ('rnfl') in s.lower() and 'symmetry' in s.lower():
                        if txts[i+1][0] == '%':
                            txts[i+1] = txts[i+1][::-1]
                        row.extend((txts[i+1],''))
                        cleaned_text.append((s, txts[i+1]))
                    if ('rim') in s.lower() and 'area' in s.lower():
                        if has_numbers(txts[i+1]) and has_numbers(txts[i+2]):
                            row.extend((txts[i+1]+'^2','', txts[i+2]+'^2',''))
                            cleaned_text.append((s, txts[i+1]+'^2', txts[i+2]+'^2'))
                    if ('disc') in s.lower() and 'area' in s.lower():
                        row.extend((txts[i+1]+'^2','', txts[i+2]+'^2',''))
                        cleaned_text.append((s, txts[i+1]+'^2', txts[i+2]+'^2'))                   
                    if ('average') in s.lower() and ('c/d' in s.lower() or 'cd' in s.lower()):
                        row.extend((txts[i+1],'', txts[i+2],''))
                        cleaned_text.append((s, txts[i+1], txts[i+2]))
                    if ('vertical') in s.lower() and ('c/d' in s.lower() or 'cd' in s.lower()):
                        row.extend((txts[i+1],'', txts[i+2],''))
                        cleaned_text.append((s, txts[i+1], txts[i+2]))
                    if ('cup') in s.lower() and 'vol' in s.lower():
                        row.extend((txts[i+1]+'^2','', txts[i+2]+'^2',''))
                        cleaned_text.append((s, txts[i+1]+'^2', txts[i+2]+'^2'))

                quadrants = ['' for i in range(8)]
                start_index, end_index = rnfl_quadrant_search_range_test(txts)
                i = 0
                for j, s in enumerate(txts[start_index:end_index]):
                    if has_numbers(s) and '%' not in s and not any(c.isalpha() for c in s):
                        quadrants[i] = s
                        i += 1
                    if 'Distribution of Normals' in s:
                        if not has_numbers(txts[start_index+j+1]):
                            i+=1
                            continue

                row.extend((quadrants[0],'',quadrants[1],''))
                cleaned_text.append(('S-OD:',quadrants[0],'S-OS:',quadrants[1]))
                row.extend((quadrants[2],'',quadrants[5],''))
                cleaned_text.append(('T-OD:',quadrants[2],'T-OS:',quadrants[5]))
                row.extend((quadrants[3],'',quadrants[4],''))
                cleaned_text.append(('N-OD:',quadrants[3],'N-OS:',quadrants[4]))
                row.extend((quadrants[6],'',quadrants[7],''))
                cleaned_text.append(('I-OD:',quadrants[6],'I-OS:',quadrants[7]))

                clock_values = ['' for i in range(24)]
                start_index, end_index = rnfl_clock_search_range_test(txts)                
                
                for i, s in enumerate(txts[start_index:end_index]):
                    if 'rnfl' in s.lower():
                        rnfl_index = i + start_index
                    if 'clock' in s.lower():
                        clock_index = i + start_index
                    if 'hours' in s.lower():
                        hours_index = i + start_index
                    if has_numbers(s) and len(s)>1 and s[0]=='0':
                        txts[start_index+i] = s[::-1]
                        
                clock_values_1 = txts[start_index:rnfl_index]
                clock_values_2 = txts[rnfl_index+1:clock_index]
                clock_values_3 = txts[clock_index+1:hours_index]
                clock_values_4 = txts[hours_index+1:]

                if len(clock_values_1)< 10:
                    clock_values_1.append('')
                if len(clock_values_2)< 4:
                    clock_values_2.append('')
                if len(clock_values_3)< 2:
                    clock_values_3.append('')
                if len(clock_values_4)< 8:
                    clock_values_4.append('')

                row.extend((clock_values_1[3], '',clock_values_1[5],''))      
                row.extend((clock_values_1[7], '',clock_values_1[9],''))      
                row.extend((clock_values_2[1],'',clock_values_2[3],''))      
                row.extend((clock_values_3[1],'',clock_values_4[1],''))      
                row.extend((clock_values_4[4],'',clock_values_4[7],''))      
                row.extend((clock_values_4[3], '',clock_values_4[6],''))      
                row.extend((clock_values_4[2], '',clock_values_4[5],''))      
                row.extend((clock_values_3[0], '',clock_values_4[0],''))      
                row.extend((clock_values_2[0], '',clock_values_2[2],''))      
                row.extend((clock_values_1[6], '',clock_values_1[8],''))      
                row.extend((clock_values_1[2], '',clock_values_1[4],'')) 
                row.extend((clock_values_1[0], '',clock_values_1[1],'')) 
                cleaned_text.append(('1-OD:', clock_values_1[3], '1-OS:',clock_values_1[5]))
                cleaned_text.append(('2-OD:', clock_values_1[7], '2-OS:',clock_values_1[9]))
                cleaned_text.append(('3-OD:', clock_values_2[1], '3-OS:',clock_values_2[3]))
                cleaned_text.append(('4-OD:', clock_values_3[1], '4-OS:',clock_values_4[1]))
                cleaned_text.append(('5-OD:', clock_values_4[4], '5-OS:',clock_values_4[7]))
                cleaned_text.append(('6-OD:', clock_values_4[3], '6-OS:',clock_values_4[6]))
                cleaned_text.append(('7-OD:', clock_values_4[2], '7-OS:',clock_values_4[5]))
                cleaned_text.append(('8-OD:', clock_values_3[0], '8-OS:',clock_values_4[0]))
                cleaned_text.append(('9-OD:', clock_values_2[0], '9-OS:',clock_values_2[2]))
                cleaned_text.append(('10-OD:', clock_values_1[6], '10-OS:',clock_values_1[8]))
                cleaned_text.append(('11-OD:', clock_values_1[2], '11-OS:',clock_values_1[4]))
                cleaned_text.append(('12-OD:', clock_values_1[0], '12-OS:',clock_values_1[1]))
                writer.writerow(row)
                # im_show = draw_ocr(image, boxes, txts, scores, font_path='/home/jupyter/PaddleOCR/doc/fonts/simfang.ttf')
                # im_show = Image.fromarray(im_show)
                # im_show.save(f'onh_rnfl/result{index}.jpg')
                                           
           
                img = Image.open(f'onh_rnfl/onh_rnfl_{index+1}.jpg')
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
                I1.text((new_width - 370, height - 600), s,font=myFont, fill=(255, 0, 0))

                # Save the edited image
                result.save(f"onh_rnfl/result{index+1}.jpg")  
                
                pdf_file_name = f'onh_rnfl/onh_rnfl_{index+1}.pdf'
                img_file_name = f'onh_rnfl/onh_rnfl_{index+1}.jpg'
                result_file_name = f"onh_rnfl/result{index+1}.jpg"
                
                destination_pdf_name = 'stanfordoptimagroup/onh_rnfl_sample_100/onh_rnfl_raw_pdf/' + pdf_file_name.split('/')[1]  
                destination_img_name = 'stanfordoptimagroup/onh_rnfl_sample_100/onh_rnfl_img/' + img_file_name.split('/')[1] 
                destination_result_name = 'stanfordoptimagroup/onh_rnfl_sample_100/onh_rnfl_result/' + result_file_name.split('/')[1] 
                    
                # upload files to bucket
                os.system(f'gsutil -o GSUtil:parallel_composite_upload_threshold=150M -m cp {pdf_file_name} gs://{destination_pdf_name}')
                os.system(f'gsutil -o GSUtil:parallel_composite_upload_threshold=150M -m cp {img_file_name} gs://{destination_img_name}')
                os.system(f'gsutil -o GSUtil:parallel_composite_upload_threshold=150M -m cp {result_file_name} gs://{destination_result_name}')
                os.system('rm -rf onh_rnfl/*')
            except:
                print('-----------')
else:
    with open('onh_rnfl_text_extraction.csv', 'a', newline ='') as csvfile:
        writer = csv.writer(csvfile, delimiter=',')

        # for index in tqdm(range(len(df_random))):
        for index in tqdm(range(len(df))):
            # dicomfilepath =df_random.iloc[index].dicomfilepath
            dicomfilepath =df.iloc[index].dicomfilepath
            source_blob_name = dicomfilepath
            output_file_name = 'onh_rnfl'+'/' + ('/').join(str(source_blob_name).split('/')[4:])
            if not os.path.exists(output_file_name):
                download_blob('stanfordoptimagroup',source_blob_name, output_file_name)
            ds = pydicom.dcmread(output_file_name,force=True)

            try:

                with open(f'onh_rnfl/onh_rnfl_{index+1}.pdf', 'wb') as fp:
                    fp.write(ds.EncapsulatedDocument)
                images = convert_from_path(f'onh_rnfl/onh_rnfl_{index+1}.pdf')

                for i in range(len(images)):
                    # Save pages as images in the pdf
                    images[i].save(f'onh_rnfl/onh_rnfl_{index+1}.jpg', 'JPEG')

                im = Image.open(f'onh_rnfl/onh_rnfl_{index+1}.jpg')
                numpydata = np.asarray(im)
                img0 = numpydata[270:310,600:1221]
                img1 = numpydata[400:790,600:1221]
                img2 = numpydata[1420:1920,600:1221]
                new_img = np.concatenate((img0,img1, img2), axis=0)
                im = Image.fromarray(new_img)
                im.save(f"onh_rnfl/onh_rnfl_{index+1}.jpg")

                ocr = PaddleOCR(use_angle_cls=True, lang='en') # need to run only once to download and load model into memory
                img_path = f'onh_rnfl/onh_rnfl_{index+1}.jpg'
                result = ocr.ocr(img_path, cls=True)
                txts = [line[1][0] for line in result]


                image = Image.open(img_path).convert('RGB')
                boxes = [line[0] for line in result]
                txts = [line[1][0] for line in result]
                scores = [line[1][1] for line in result]
                #-------------------
                row = []
                cleaned_text = []

                row.append(f'onh_rnfl_{index+1}')
                for i, s in enumerate(txts):
                    if ('signal strength') in s.lower():
                        r1 = re.findall(r"\d/10",s)

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
                for i, s in enumerate(txts):
                    if ('average') in s.lower() and ('rnfl') in s.lower()  and ('thickness') in s.lower():
                        ct = 0

                        l = []
                        for j, s in enumerate(txts[i:i+4]):
                            if 'mm' in s.lower():
                                continue
                            if 'm' in s.lower() and len(s)<=5:
                                ct+=1

                                tmp_s = ''
                                for c in s:
                                    if c == 'm':

                                        tmp_s+=('\u03BC'+ c)
                                    else:

                                        tmp_s+=c
                                l.extend([tmp_s, ''])
  
                        row.extend(l)
                        cleaned_text.append(('average rnfl thickness', l[0]))
                        cleaned_text.append(('average rnfl thickness', l[2]))
                        if ct == 1:
                            row.extend(['',''])
                            
                for i, s in enumerate(txts):
                    if ('rnfl') in s.lower() and 'symmetry' in s.lower():
                        if txts[i+1][0] == '%':
                            txts[i+1] = txts[i+1][::-1]
                        row.extend((txts[i+1],''))
                        cleaned_text.append((s, txts[i+1]))
                    if ('rim') in s.lower() and 'area' in s.lower():
                        if has_numbers(txts[i+1]) and has_numbers(txts[i+2]):
                            row.extend((txts[i+1]+'^2','', txts[i+2]+'^2',''))
                            cleaned_text.append((s, txts[i+1]+'^2', txts[i+2]+'^2'))
                    if ('disc') in s.lower() and 'area' in s.lower():
                        row.extend((txts[i+1]+'^2','', txts[i+2]+'^2',''))
                        cleaned_text.append((s, txts[i+1]+'^2', txts[i+2]+'^2'))                   
                    if ('average') in s.lower() and ('c/d' in s.lower() or 'cd' in s.lower()):
                        row.extend((txts[i+1],'', txts[i+2],''))
                        cleaned_text.append((s, txts[i+1], txts[i+2]))
                    if ('vertical') in s.lower() and ('c/d' in s.lower() or 'cd' in s.lower()):
                        row.extend((txts[i+1],'', txts[i+2],''))
                        cleaned_text.append((s, txts[i+1], txts[i+2]))
                    if ('cup') in s.lower() and 'vol' in s.lower():
                        row.extend((txts[i+1]+'^2','', txts[i+2]+'^2',''))
                        cleaned_text.append((s, txts[i+1]+'^2', txts[i+2]+'^2'))

                quadrants = ['' for i in range(8)]
                start_index, end_index = rnfl_quadrant_search_range_test(txts)
                i = 0
                for j, s in enumerate(txts[start_index:end_index]):
                    if has_numbers(s) and '%' not in s and not any(c.isalpha() for c in s):
                        quadrants[i] = s
                        i += 1
                    if 'Distribution of Normals' in s:
                        if not has_numbers(txts[start_index+j+1]):
                            i+=1
                            continue

                row.extend((quadrants[0],'',quadrants[1],''))
                cleaned_text.append(('S-OD:',quadrants[0],'S-OS:',quadrants[1]))
                row.extend((quadrants[2],'',quadrants[5],''))
                cleaned_text.append(('T-OD:',quadrants[2],'T-OS:',quadrants[5]))
                row.extend((quadrants[3],'',quadrants[4],''))
                cleaned_text.append(('N-OD:',quadrants[3],'N-OS:',quadrants[4]))
                row.extend((quadrants[6],'',quadrants[7],''))
                cleaned_text.append(('I-OD:',quadrants[6],'I-OS:',quadrants[7]))

                clock_values = ['' for i in range(24)]
                start_index, end_index = rnfl_clock_search_range_test(txts)                
                
                for i, s in enumerate(txts[start_index:end_index]):
                    if 'rnfl' in s.lower():
                        rnfl_index = i + start_index
                    if 'clock' in s.lower():
                        clock_index = i + start_index
                    if 'hours' in s.lower():
                        hours_index = i + start_index
                    if has_numbers(s) and len(s)>1 and s[0]=='0':
                        txts[start_index+i] = s[::-1]
                        
                clock_values_1 = txts[start_index:rnfl_index]
                clock_values_2 = txts[rnfl_index+1:clock_index]
                clock_values_3 = txts[clock_index+1:hours_index]
                clock_values_4 = txts[hours_index+1:]

                if len(clock_values_1)< 10:
                    clock_values_1.append('')
                if len(clock_values_2)< 4:
                    clock_values_2.append('')
                if len(clock_values_3)< 2:
                    clock_values_3.append('')
                if len(clock_values_4)< 8:
                    clock_values_4.append('')

                row.extend((clock_values_1[3], '',clock_values_1[5],''))      
                row.extend((clock_values_1[7], '',clock_values_1[9],''))      
                row.extend((clock_values_2[1],'',clock_values_2[3],''))      
                row.extend((clock_values_3[1],'',clock_values_4[1],''))      
                row.extend((clock_values_4[4],'',clock_values_4[7],''))      
                row.extend((clock_values_4[3], '',clock_values_4[6],''))      
                row.extend((clock_values_4[2], '',clock_values_4[5],''))      
                row.extend((clock_values_3[0], '',clock_values_4[0],''))      
                row.extend((clock_values_2[0], '',clock_values_2[2],''))      
                row.extend((clock_values_1[6], '',clock_values_1[8],''))      
                row.extend((clock_values_1[2], '',clock_values_1[4],'')) 
                row.extend((clock_values_1[0], '',clock_values_1[1],'')) 
                cleaned_text.append(('1-OD:', clock_values_1[3], '1-OS:',clock_values_1[5]))
                cleaned_text.append(('2-OD:', clock_values_1[7], '2-OS:',clock_values_1[9]))
                cleaned_text.append(('3-OD:', clock_values_2[1], '3-OS:',clock_values_2[3]))
                cleaned_text.append(('4-OD:', clock_values_3[1], '4-OS:',clock_values_4[1]))
                cleaned_text.append(('5-OD:', clock_values_4[4], '5-OS:',clock_values_4[7]))
                cleaned_text.append(('6-OD:', clock_values_4[3], '6-OS:',clock_values_4[6]))
                cleaned_text.append(('7-OD:', clock_values_4[2], '7-OS:',clock_values_4[5]))
                cleaned_text.append(('8-OD:', clock_values_3[0], '8-OS:',clock_values_4[0]))
                cleaned_text.append(('9-OD:', clock_values_2[0], '9-OS:',clock_values_2[2]))
                cleaned_text.append(('10-OD:', clock_values_1[6], '10-OS:',clock_values_1[8]))
                cleaned_text.append(('11-OD:', clock_values_1[2], '11-OS:',clock_values_1[4]))
                cleaned_text.append(('12-OD:', clock_values_1[0], '12-OS:',clock_values_1[1]))
                writer.writerow(row)
                # im_show = draw_ocr(image, boxes, txts, scores, font_path='/home/jupyter/PaddleOCR/doc/fonts/simfang.ttf')
                # im_show = Image.fromarray(im_show)
                # im_show.save(f'onh_rnfl/result{index}.jpg')
                #-------------------
                
                
                
                #-------------------
                                           
                img = Image.open(f'onh_rnfl/onh_rnfl_{index+1}.jpg')
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
                I1.text((new_width - 370, height - 600), s,font=myFont, fill=(255, 0, 0))

                # Save the edited image
                result.save(f"onh_rnfl/result{index+1}.jpg")  
                #-------------------
                
                pdf_file_name = f'onh_rnfl/onh_rnfl_{index+1}.pdf'
                img_file_name = f'onh_rnfl/onh_rnfl_{index+1}.jpg'
                result_file_name = f"onh_rnfl/result{index+1}.jpg"
                
                destination_pdf_name = 'stanfordoptimagroup/onh_rnfl_sample_100/onh_rnfl_raw_pdf/' + pdf_file_name.split('/')[1]  
                destination_img_name = 'stanfordoptimagroup/onh_rnfl_sample_100/onh_rnfl_img/' + img_file_name.split('/')[1] 
                destination_result_name = 'stanfordoptimagroup/onh_rnfl_sample_100/onh_rnfl_result/' + result_file_name.split('/')[1] 
                    
                # upload files to bucket
                os.system(f'gsutil -o GSUtil:parallel_composite_upload_threshold=150M -m cp {pdf_file_name} gs://{destination_pdf_name}')
                os.system(f'gsutil -o GSUtil:parallel_composite_upload_threshold=150M -m cp {img_file_name} gs://{destination_img_name}')
                os.system(f'gsutil -o GSUtil:parallel_composite_upload_threshold=150M -m cp {result_file_name} gs://{destination_result_name}')
                os.system('rm -rf onh_rnfl/*')
                
            except:
                print('-----------')
upload_blob('stanfordoptimagroup', 'onh_rnfl_text_extraction.csv', 'onh_rnfl_text_extraction.csv',verbose=False)
onh_rnfl_text_extraction = pd.read_csv('onh_rnfl_text_extraction.csv')
onh_rnfl_text_extraction = pyarrow_dtype_format_correction(onh_rnfl_text_extraction)
saved_table_id = 'imaging'+f".onh_rnfl_text_extraction"
onh_rnfl_text_extraction.to_gbq(saved_table_id, project_id = 'som-nero-phi-sywang-starr', if_exists = 'replace') 
