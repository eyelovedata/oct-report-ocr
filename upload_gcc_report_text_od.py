from utils import *
project_id = 'som-nero-phi-sywang-starr'
dataset_id = 'imaging'
table_id = 'EncapsulatedDocument_all_3_batches'
fieldnames =['gcc_report_number',
            'MRN',
            'Exam_Date',
            'signal_strength_OD',
            'S_OD',
            'SN_OD',
            'IN_OD',
            'I_OD',
            'IT_OD',
            'ST_OD',
            'Average_GCL_IPLThickness_OD',
            'Minimum_GCL_IPLThickness_OD']



query="""SELECT * FROM `som-nero-phi-sywang-starr.imaging.EncapsulatedDocument_all_3_batches` 
where DocumentTitle like "%OD Ganglion Cell OU Analysis%"
"""
query_job =client.query(query)
df=query_job.to_dataframe()
df.columns = map(str.lower, df.columns)

if not os.path.exists('gcc_text_extraction_od.csv'):
    with open('gcc_text_extraction_od.csv', 'w', newline ='') as csvfile:
        writer = csv.writer(csvfile, delimiter=',')
        writer.writerow(fieldnames)

        for index in tqdm(range(len(df))):
            dicomfilepath =df.iloc[index].dicomfilepath
            source_blob_name = dicomfilepath
            output_file_name = 'gcc'+'/' + ('/').join(str(source_blob_name).split('/')[4:])
            if not os.path.exists(output_file_name):
                download_blob('stanfordoptimagroup',source_blob_name, output_file_name)
            ds = pydicom.dcmread(output_file_name,force=True)

            try:

                with open(f'gcc/gcc_od_{index+1}.pdf', 'wb') as fp:
                    fp.write(ds.EncapsulatedDocument)
                images = convert_from_path(f'gcc/gcc_od_{index+1}.pdf')

                for i in range(len(images)):
                    # Save pages as images in the pdf
                    images[i].save(f'gcc/gcc_od_{index+1}.jpg', 'JPEG')

                im = Image.open(f'gcc/gcc_od_{index+1}.jpg')
                numpydata = np.asarray(im)
                img0 = numpydata[270:315,610:1120]
                img1 = np.concatenate((numpydata[1025:1240,555:810],np.full((20,255,3),255),numpydata[1025:1240,995:1250]), axis=0)
                img2 = np.concatenate((img1,np.full((450,255,3), 255)), axis=1)
                img3 = numpydata[1300:1400,650:1160]
                new_img = np.concatenate((img0,img2,np.full((20,510,3),255),img3), axis=0)

                im = Image.fromarray(np.uint8(new_img))
                im.save(f"gcc/gcc_od_{index+1}.jpg")

                ocr = PaddleOCR(use_angle_cls=True, lang='en') # need to run only once to download and load model into memory
                img_path = f'gcc/gcc_od_{index+1}.jpg'
                result = ocr.ocr(img_path, cls=True)
                txts = [line[1][0] for line in result]


                image = Image.open(img_path).convert('RGB')
                boxes = [line[0] for line in result]
                txts = [line[1][0] for line in result]
                scores = [line[1][1] for line in result]
                row = []
                
                row.append(f'gcc_od_{index+1}')
                row.extend((ds.PatientID, ds.StudyDate))
                for i, s in enumerate(txts):
                    if ('signal strength') in s.lower():
                        r1 = re.findall(r"\d{1,2}/10",s)
                        if r1 == []:
                            row.append((txts[i+1]))
                        else:
                            row.append((r1[0]))
                
                quadrants = ['' for i in range(6)]                              
                start_index, end_index = gcc_search_range(txts)
                i = 0
                for j, s in enumerate(txts[start_index:end_index]):
                    if i<6:
                        quadrants[i] = s
                        i += 1                     
                        
                row.append((quadrants[0]))
                row.append((quadrants[2]))
                row.append((quadrants[4]))
                row.append((quadrants[5]))
                row.append((quadrants[3]))
                row.append((quadrants[1]))
                for i, s in enumerate(txts):
                    if ('ave') in s.lower():
                        row.append(((txts[i+1] if has_numbers(txts[i+1]) else txts[i+2])))
                    if ('min') in s.lower():
                        row.append(((txts[i+1] if has_numbers(txts[i+1]) else txts[i+2])))
                writer.writerow(row)
                os.system('rm -rf gcc/*')
            except:
                print('-----------')
upload_blob('stanfordoptimagroup', 'gcc_text_extraction_od.csv', 'gcc_text_extraction_od.csv',verbose=False)
gcc_text_extraction = pd.read_csv('gcc_text_extraction_od.csv')
gcc_text_extraction = pyarrow_dtype_format_correction(gcc_text_extraction)
saved_table_id = 'imaging'+f".gcc_text_extraction_od"
gcc_text_extraction.to_gbq(saved_table_id, project_id = 'som-nero-phi-sywang-starr', if_exists = 'replace') 
