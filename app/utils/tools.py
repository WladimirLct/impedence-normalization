# File: utils/tools.py

def format_group_name(group):
    """Format group name as 'Name (Concentration Unit)'"""
    name = group.get('name', 'Unknown')
    concentration = group.get('concentration', '')
    unit = group.get('unit', '')
    
    if concentration and unit:
        return f"{name} ({concentration} {unit})"
    else:
        return name