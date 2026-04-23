def celsius_to_fahrenheit(celsius):
    """Convert a temperature from Celsius to Fahrenheit."""
    fahrenheit = (celsius * 9/5) + 32
    return fahrenheit

def fahrenheit_to_celsius(fahrenheit):
    """Convert a temperature from Fahrenheit to Celsius."""
    celsius = (fahrenheit - 32) * 5/9
    return celsius

def meters_to_feet(meters):
    """Convert a length from meters to feet."""
    feet = meters * 3.28084
    return feet

def feet_to_meters(feet):
    """Convert a length from feet to meters."""
    meters = feet / 3.28084
    return meters

def mm_to_inches(mm):
    """Convert a length from millimeters to inches."""
    inches = mm / 25.4
    return inches

def inches_to_mm(inches):
    """Convert a length from inches to millimeters."""
    mm = inches * 25.4
    return mm

def kg_to_pounds(kg):
    """Convert a mass from kilograms to pounds."""
    pounds = kg * 2.20462
    return pounds

def pounds_to_kg(pounds):
    """Convert a mass from pounds to kilograms."""
    kg = pounds / 2.20462
    return kg

def cubic_meters_to_cubic_feet(cubic_meters):
    """Convert a volume from cubic meters to cubic feet."""
    cubic_feet = cubic_meters * 35.3147
    return cubic_feet

def cubic_feet_to_cubic_meters(cubic_feet):
    """Convert a volume from cubic feet to cubic meters."""
    cubic_meters = cubic_feet / 35.3147
    return cubic_meters