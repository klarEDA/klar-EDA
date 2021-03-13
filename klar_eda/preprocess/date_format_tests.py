from .csv_preprocess import CSVPreProcess


def test_convert_date_format():
	file_path = "" #add path to your test data
	csvPreP = CSVPreProcess(file_path)

	csvPreP.convert_date_format('2021/28/02', 'YYYY-MM-DD')
	csvPreP.convert_date_format('02/28/2021', 'YYYY/DD/MM')
	csvPreP.convert_date_format('2021/02/28', 'MM-DD-YYYY')

	csvPreP.convert_date_format('28-02-2021', 'YYYY-MM-DD')
	csvPreP.convert_date_format('2021-28-02', 'MM/DD/YYYY')
	csvPreP.convert_date_format('02-28-2021')
	csvPreP.convert_date_format('2021-02-28', 'YYYY/MM/DD')

	csvPreP.convert_date_format('2021/12/03', 'YYYY-MM-DD')
	csvPreP.convert_date_format('03/12/2021', 'YYYY/DD/MM')
	csvPreP.convert_date_format('2021/03/12', 'MM-DD-YYYY')

	csvPreP.convert_date_format('12-03-2021', 'YYYY-MM-DD')
	csvPreP.convert_date_format('2021-12-03', 'MM/DD/YYYY')
	csvPreP.convert_date_format('03-12-2021')
	csvPreP.convert_date_format('2021-03-12', 'YYYY/MM/DD')




 