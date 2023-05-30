from utils import *
project_id = 'som-nero-phi-sywang-starr'
dataset_id = 'imaging'
table_id = 'EncapsulatedDocument_all_3_batches'
fieldnames =['onh_rnfl_report_number',
            'MRN',
            'Exam_Date',
            'signal_strength_OD',
            'average_rnfl_thickness_OD',
            'Rim_Area_OD',
            'Disc_Area_OD',
            'Average_C_D_Ratio_OD',
            'Vertical_C_D_Ratio_OD',
            'Cup_Volume_OD',
            'S_OD',
            'T_OD',
            'N_OD',
            'I_OD']

query="""SELECT *
FROM `{project_id}.{dataset_id}.{table_id}`where DocumentTitle like '%Cirrus_OD_ONH and RNFL OU Analysis%'
 """.format_map({'project_id': project_id,
                'dataset_id': dataset_id,
                'table_id': table_id})
query_job =client.query(query)
df=query_job.to_dataframe()
df.columns = map(str.lower, df.columns)

# df_random = df.sample(n=2, random_state=3)
if not os.path.exists('onh_rnfl_text_extraction_od.csv'):
    with open('onh_rnfl_text_extraction_od.csv', 'w', newline ='') as csvfile:
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

                with open(f'onh_rnfl/onh_rnfl_od_{index+1}.pdf', 'wb') as fp:
                    fp.write(ds.EncapsulatedDocument)
                images = convert_from_path(f'onh_rnfl/onh_rnfl_od_{index+1}.pdf')

                for i in range(len(images)):
                    # Save pages as images in the pdf
                    images[i].save(f'onh_rnfl/onh_rnfl_od_{index+1}.jpg', 'JPEG')

                im = Image.open(f'onh_rnfl/onh_rnfl_od_{index+1}.jpg')
                numpydata = np.asarray(im)
                img0 = numpydata[270:310,600:1221]
                img1 = numpydata[450:790,600:1221]
                img2 = numpydata[1420:1920,600:1221]

                new_img = np.concatenate((img0,img1,img2), axis=0)
                im = Image.fromarray(new_img)
                im.save(f"onh_rnfl/onh_rnfl_od_{index+1}.jpg")

                ocr = PaddleOCR(use_angle_cls=True, lang='en') # need to run only once to download and load model into memory
                img_path = f'onh_rnfl/onh_rnfl_od_{index+1}.jpg'
                result = ocr.ocr(img_path, cls=True)
                txts = [line[1][0] for line in result]


                image = Image.open(img_path).convert('RGB')
                boxes = [line[0] for line in result]
                txts = [line[1][0] for line in result]
                scores = [line[1][1] for line in result]
                row = []
                
                row.append(f'onh_rnfl_od_{index+1}')
                row.extend((ds.PatientID, ds.StudyDate))
                for i, s in enumerate(txts):
                    if ('signal strength') in s.lower():
                        r1 = re.findall(r"\d{1,2}/10",s)

                        if r1 == []:
                            row.append((txts[i+1]))                                
                        else:
                            row.append((r1[0]))
                for i, s in enumerate(txts):
                    if ('average') in s.lower() and ('rnfl') in s.lower()  and ('thickness') in s.lower():
                        ct = 0

                        l = []
                        for j, s in enumerate(txts[i:i+4]):
                            if 'mm' in s.lower():
                                continue
                            if 'm' in s.lower() and len(s)<=5:
                                tmp_s = ''
                                for c in s:
                                    if c == 'm':

                                        tmp_s+=('\u03BC'+ c)
                                    else:
                                        tmp_s+=c
                                l.extend([tmp_s])
                        row.append(l[0])
                        
                            
                for i, s in enumerate(txts):
                    if ('rim') in s.lower() and 'area' in s.lower():
                        row.append(((txts[i+1]+'^2' if has_numbers(txts[i+1]) else txts[i+2]+'^2')))
                    if ('disc') in s.lower() and 'area' in s.lower():
                        row.append(((txts[i+1]+'^2' if has_numbers(txts[i+1]) else txts[i+2]+'^2')))
                    if ('average') in s.lower() and ('c/d' in s.lower() or 'cd' in s.lower()):
                        row.append(((txts[i+1] if has_numbers(txts[i+1]) else txts[i+2])))
                    if ('vertical') in s.lower() and ('c/d' in s.lower() or 'cd' in s.lower()):
                        row.append(((txts[i+1] if has_numbers(txts[i+1]) else txts[i+2])))
                    if ('cup') in s.lower() and 'vol' in s.lower():
                        row.append(((txts[i+1]+'^2' if has_numbers(txts[i+1]) else txts[i+2]+'^2')))
                            
                
                quadrants = ['' for i in range(4)]
                start_index, end_index = rnfl_quadrant_search_range_test(txts)
                i = 0
                for j, s in enumerate(txts[start_index:end_index]):
                    if i<4 and has_numbers(s) and '%' not in s and not any(c.isalpha() for c in s):
                        quadrants[i] = s
                        i += 1 
                        
                row.append((quadrants[0]))
                row.append((quadrants[1]))
                row.append((quadrants[2]))
                row.append((quadrants[3]))
                writer.writerow(row)
                os.system('rm -rf onh_rnfl/*')
            except:
                 print('-----------')
upload_blob('stanfordoptimagroup', 'onh_rnfl_text_extraction_od.csv', 'onh_rnfl_text_extraction_od.csv',verbose=False)
onh_rnfl_text_extraction = pd.read_csv('onh_rnfl_text_extraction_od.csv')
onh_rnfl_text_extraction = pyarrow_dtype_format_correction(onh_rnfl_text_extraction)
saved_table_id = 'imaging'+f".onh_rnfl_text_extraction_od"
onh_rnfl_text_extraction.to_gbq(saved_table_id, project_id = 'som-nero-phi-sywang-starr', if_exists = 'replace') 
