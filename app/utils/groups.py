# File: utils/groups.py

import os 
from config.cache import cache

def obj_wells_groups(name, concentration, label, unit, wells):
    return {
        "name": name,
        "concentration": concentration,
        "label": label,
        "unit": unit,
        "wells": wells
    }

def get_list_wells():
    if "wells" not in cache:
        cache["wells"] = []
    return cache["wells"]

def get_list_wells_groups():
    if "well_groups" not in cache:
        cache["well_groups"] = []
    return cache["well_groups"]

def add_wells_groups(groups_list, new_group):
    if not isinstance(groups_list, list):
        raise ValueError("groups_list must be a list")
    
    if not isinstance(new_group, dict):
        raise ValueError("new_group must be a dictionary")
    
    groups_list.append(new_group)
    return groups_list
    
def create_single_well_groups(wells):
    if len(wells) < 1:
        raise ValueError("At least one well is required to create a group")

    group_list = []

    for well in wells:
        group_list.append(obj_wells_groups(
            name=well,
            concentration=-1,  
            label="",  
            unit="",
            wells=[well]
        ))
    return group_list

def get_wells_not_in_group(wells, groups):

    if not isinstance(wells, list) or not isinstance(groups, list):
        raise ValueError("Invalid input types")
    wells_in_groups = set()
    for group in groups:
        wells_in_groups.update(group.get("wells", []))
    return [well for well in wells if well not in wells_in_groups]