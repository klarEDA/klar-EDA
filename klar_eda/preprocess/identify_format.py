import datetime
import dateutil.parser


def convert_date_format(input_date, output_date_format = 'DD/MM/YYYY'):
	"""
	Purpose:
	---
	Converts the input date into any specified format

	Input Arguments:
	---
	`input_date` : str

	`output_date_format` : str

	Returns:
	---
	`converted_date` : str
	
	Example call:
	---
	convert_date_format('2021/28/02', 'YYYY-MM-DD')

	NOTE: This function will be automatically called by test_task_2a executable at regular intervals.
	"""

	output_date_formats = { 'DD/MM/YYYY' : '%d/%m/%Y', 'YYYY/DD/MM': '%Y/%d/%m', 'MM/DD/YYYY': '%m/%d/%Y',       \
							'YYYY/MM/DD' : '%Y/%m/%d', 'DD-MM-YYYY': '%d-%m-%Y', 'YYYY-DD-MM': '%Y-%d-%m',       \
							'MM-DD-YYYY' : '%m-%d-%Y', 'YYYY-MM-DD': '%Y-%m-%d'}
	
	
	parsed_date = dateutil.parser.parse(input_date, dayfirst=True)
	converted_date = parsed_date.strftime(output_date_formats[output_date_format])
	print(converted_date)
	return converted_date


if __name__ == "__main__":


	# Tested on the following use cases

	convert_date_format('2021/28/02', 'YYYY-MM-DD')
	convert_date_format('02/28/2021', 'YYYY/DD/MM')
	convert_date_format('2021/02/28', 'MM-DD-YYYY')

	convert_date_format('28-02-2021', 'YYYY-MM-DD')
	convert_date_format('2021-28-02', 'MM/DD/YYYY')
	convert_date_format('02-28-2021')
	convert_date_format('2021-02-28', 'YYYY/MM/DD')

	convert_date_format('2021/12/03', 'YYYY-MM-DD')
	convert_date_format('03/12/2021', 'YYYY/DD/MM')
	convert_date_format('2021/03/12', 'MM-DD-YYYY')

	convert_date_format('12-03-2021', 'YYYY-MM-DD')
	convert_date_format('2021-12-03', 'MM/DD/YYYY')
	convert_date_format('03-12-2021')
	convert_date_format('2021-03-12', 'YYYY/MM/DD')


 