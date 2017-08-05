from app.NLP_LDA import *

def test_read_letters():
    letters = read_letter_from_file("test\example_shareholders_letter.csv")
    assert len(letters) == 38

def test_read_company():
    company = read_letter_from_file("test\example_shareholders_letter.csv")
    assert len(company) == 38

def test_number_of_company():
    company = read_company_from_file("test\example_shareholders_letter.csv")
    number_of_company = len(set(company))
    assert number_of_company == 5

def test_name_of_company():
    company = read_company_from_file("test\example_shareholders_letter.csv")
    name_of_company = set(company)
    assert name_of_company == {'Aspen ', 'Adobe ', 'ea ', 'Citrix ', 'Compuware '}
