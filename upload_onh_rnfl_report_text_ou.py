from utils import *
project_id = 'som-nero-phi-sywang-starr'
dataset_id = 'imaging'
table_id = 'EncapsulatedDocument_all_3_batches'
fieldnames =['onh_rnfl_report_number',
            'MRN',
            'Exam_Date',
            'signal_strength_OD',
            'signal_strength_OS',
            'average_rnfl_thickness_OD',
            'average_rnfl_thickness_OS',
            'RNFL_Symmetry',
            'Rim_Area_OD',
            'Rim_Area_OS',
            'Disc_Area_OD',
            'Disc_Area_OS',
            'Average_CD_Ratio_OD',
            'Average_CD_Ratio_OS',
            'Vertical_CD_Ratio_OD',
            'Vertical_CD_Ratio_OS',
            'Cup_Volume_OD',
            'Cup_Volume_OS',
            'S_OD',
            'S_OS',
            'T_OD',
            'T_OS',
            'N_OD',
            'N_OS',
            'I_OD',
            'I_OS',
            't1_OD',
            't1_OS',
            't2_OD',
            't2_OS',
            't3_OD',
            't3_OS',
            't4_OD',
            't4_OS',
            't5_OD',
            't5_OS',
            't6_OD',
            't6_OS',
            't7_OD',
            't7_OS',
            't8_OD',
            't8_OS',
            't9_OD',
            't9_OS',
            't10_OD',
            't10_OS',
            't11_OD',
            't11_OS',
            't12_OD',
            't12_OS']

query="""SELECT *
FROM `{project_id}.{dataset_id}.{table_id}`where DocumentTitle like '%Cirrus_OU_ONH and RNFL OU Analysis%'
 """.format_map({'project_id': project_id,
                'dataset_id': dataset_id,
                'table_id': table_id})
query_job =client.query(query)
df=query_job.to_dataframe()
df.columns = map(str.lower, df.columns)

if not os.path.exists('onh_rnfl_text_extraction_ou.csv'):
    with open('onh_rnfl_text_extraction_ou.csv', 'w', newline ='') as csvfile:
        writer = csv.writer(csvfile, delimiter=',')
        writer.writerow(fieldnames)

        for index in tqdm(range(len(df))):
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
                
                row.append(f'onh_rnfl_{index+1}')
                row.extend((ds.PatientID, ds.StudyDate))
                for i, s in enumerate(txts):
                    if ('signal strength') in s.lower():
                        r1 = re.findall(r"\d{1,2}/10",s)

                        if r1 == []:
                            if i == 0:
                                row.extend((txts[i+1],txts[i+2]))
                        
                            if i == 1:
                                row.extend((txts[i+1],txts[i-1]))
                        
                                
                        elif len(r1) == 1:
                            if i == 0:
                                row.extend((r1[0], txts[i+1]))
                        
                            if i == 1:
                                row.extend((r1[0], txts[i-1]))
                        
                        else:
                            row.extend((r1[0],r1[1]))
                        
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
                                l.extend([tmp_s])
  
                        row.extend(l)

                        if ct == 1:
                            row.extend([''])
                            
                for i, s in enumerate(txts):
                    if ('rnfl') in s.lower() and 'symmetry' in s.lower():
                        if txts[i+1][0] == '%':
                            txts[i+1] = txts[i+1][::-1]
                        row.append((txts[i+1]))

                    if ('rim') in s.lower() and 'area' in s.lower():
                        if has_numbers(txts[i+1]) and has_numbers(txts[i+2]):
                            row.extend((txts[i+1]+'^2', txts[i+2]+'^2'))

                    if ('disc') in s.lower() and 'area' in s.lower():
                        row.extend((txts[i+1]+'^2', txts[i+2]+'^2'))
                  
                    if ('average') in s.lower() and ('c/d' in s.lower() or 'cd' in s.lower()):
                        row.extend((txts[i+1], txts[i+2]))

                    if ('vertical') in s.lower() and ('c/d' in s.lower() or 'cd' in s.lower()):
                        row.extend((txts[i+1], txts[i+2]))

                    if ('cup') in s.lower() and 'vol' in s.lower():
                        row.extend((txts[i+1]+'^2', txts[i+2]+'^2'))


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

                row.extend((quadrants[0],quadrants[1]))
                row.extend((quadrants[2],quadrants[5]))
                row.extend((quadrants[3],quadrants[4]))
                row.extend((quadrants[6],quadrants[7]))


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

                row.extend((clock_values_1[3],clock_values_1[5]))      
                row.extend((clock_values_1[7],clock_values_1[9]))      
                row.extend((clock_values_2[1],clock_values_2[3]))      
                row.extend((clock_values_3[1],clock_values_4[1]))      
                row.extend((clock_values_4[4],clock_values_4[7]))      
                row.extend((clock_values_4[3],clock_values_4[6]))      
                row.extend((clock_values_4[2],clock_values_4[5]))      
                row.extend((clock_values_3[0],clock_values_4[0]))      
                row.extend((clock_values_2[0],clock_values_2[2]))      
                row.extend((clock_values_1[6],clock_values_1[8]))      
                row.extend((clock_values_1[2],clock_values_1[4])) 
                row.extend((clock_values_1[0],clock_values_1[1])) 
                writer.writerow(row)                                           
                os.system('rm -rf onh_rnfl/*')
            except:
                print('-----------')
else:
    with open('onh_rnfl_text_extraction_ou.csv', 'a', newline ='') as csvfile:
        writer = csv.writer(csvfile, delimiter=',')

        for index in tqdm(range(len(df))):
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
                row.append(f'onh_rnfl_{index+1}')
                row.extend((ds.PatientID, ds.StudyDate))
                for i, s in enumerate(txts):
                    if ('signal strength') in s.lower():
                        r1 = re.findall(r"\d/10",s)

                        if r1 == []:
                            if i == 0:
                                row.extend((txts[i+1],txts[i+2]))
                                
                            if i == 1:
                                row.extend((txts[i+1],txts[i-1]))
                               
                                
                        elif len(r1) == 1:
                            if i == 0:
                                row.extend((r1[0], txts[i+1]))
                               
                            if i == 1:
                                row.extend((r1[0], txts[i-1]))
                               
                        else:
                            row.extend((r1[0],r1[1]))
                           
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
                                l.extend([tmp_s])
  
                        row.extend(l)
                        if ct == 1:
                            row.extend([''])
                            
                for i, s in enumerate(txts):
                    if ('rnfl') in s.lower() and 'symmetry' in s.lower():
                        if txts[i+1][0] == '%':
                            txts[i+1] = txts[i+1][::-1]
                        row.append((txts[i+1]))

                    if ('rim') in s.lower() and 'area' in s.lower():
                        if has_numbers(txts[i+1]) and has_numbers(txts[i+2]):
                            row.extend((txts[i+1]+'^2', txts[i+2]+'^2'))
                    if ('disc') in s.lower() and 'area' in s.lower():
                        row.extend((txts[i+1]+'^2', txts[i+2]+'^2'))                 
                    if ('average') in s.lower() and ('c/d' in s.lower() or 'cd' in s.lower()):
                        row.extend((txts[i+1], txts[i+2]))
                    if ('vertical') in s.lower() and ('c/d' in s.lower() or 'cd' in s.lower()):
                        row.extend((txts[i+1], txts[i+2]))
                    if ('cup') in s.lower() and 'vol' in s.lower():
                        row.extend((txts[i+1]+'^2', txts[i+2]+'^2'))

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

                row.extend((quadrants[0],quadrants[1]))
                row.extend((quadrants[2],quadrants[5]))
                row.extend((quadrants[3],quadrants[4]))
                row.extend((quadrants[6],quadrants[7]))

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

                row.extend((clock_values_1[3],clock_values_1[5]))      
                row.extend((clock_values_1[7],clock_values_1[9]))      
                row.extend((clock_values_2[1],clock_values_2[3]))      
                row.extend((clock_values_3[1],clock_values_4[1]))      
                row.extend((clock_values_4[4],clock_values_4[7]))      
                row.extend((clock_values_4[3],clock_values_4[6]))      
                row.extend((clock_values_4[2],clock_values_4[5]))      
                row.extend((clock_values_3[0],clock_values_4[0]))      
                row.extend((clock_values_2[0],clock_values_2[2]))      
                row.extend((clock_values_1[6],clock_values_1[8]))      
                row.extend((clock_values_1[2],clock_values_1[4])) 
                row.extend((clock_values_1[0],clock_values_1[1])) 
                writer.writerow(row)
                os.system('rm -rf onh_rnfl/*')
                
            except:
                print('-----------')
upload_blob('stanfordoptimagroup', 'onh_rnfl_text_extraction_ou.csv', 'onh_rnfl_text_extraction_ou.csv',verbose=False)
onh_rnfl_text_extraction = pd.read_csv('onh_rnfl_text_extraction_ou.csv')
onh_rnfl_text_extraction = pyarrow_dtype_format_correction(onh_rnfl_text_extraction)
saved_table_id = 'imaging'+f".onh_rnfl_text_extraction_ou"
onh_rnfl_text_extraction.to_gbq(saved_table_id, project_id = 'som-nero-phi-sywang-starr', if_exists = 'replace') 
