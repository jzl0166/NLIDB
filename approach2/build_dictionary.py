import numpy as np
all_dicts = {}

subset = 'basketball'
subset_dict = {}
subset_dict['number_of_assists']=['2', '3', '4']
subset_dict['number_of_blocks']=['2', '3', '4']
subset_dict['number_of_turnovers']=['2', '3', '4']
subset_dict['number_of_points']=['2', '3', '4']
subset_dict['number_of_fouls']=['2', '3', '4']
subset_dict['number_of_steals']=['2', '3', '4']
subset_dict['number_of_played_games']=['2', '3', '4']
subset_dict['player']=['lebron_james', 'kobe_bryant']
subset_dict['team']=['los_angeles_lakers','cleveland_cavaliers']
subset_dict['position']=['point_guard','forward']
subset_dict['season']=['2004', '2010']
all_dicts[subset] = subset_dict

#------------------------------------------------------
subset = 'calendar'
subset_dict = {}
subset_dict['location']=['central_office', 'greenberg_cafe']
subset_dict['important']=['true']
subset_dict['meeting']=['annual_review', 'weekly_standup']
subset_dict['attendee']=['alice', 'bob']
subset_dict['length']=['one', 'two', 'three']
subset_dict['end_time']=['1pm', '3pm', '10am']
subset_dict['start_time']=['1pm', '3pm', '10am']
subset_dict['date']=['jan_2nd', 'jan_3rd']
all_dicts[subset] = subset_dict

#------------------------------------------------------
subset = 'housing'
subset_dict = {}
subset_dict['neighborhood']=['midtown_west', 'chelsea']
subset_dict['housing_type']=['flat', 'apartment','condo']
subset_dict['housing_unit']=['123_sesame_street', '900_mission_avenue']
subset_dict['posting_date']=['feb_3rd', 'jan_2nd']
subset_dict['size']=['800', '1000']
subset_dict['monthly_rent']=['1500', '2000']
subset_dict['cats']=['true']
subset_dict['dogs']=['true']
subset_dict['private_bath']=['true']
subset_dict['private_room']=['true']    
all_dicts[subset] = subset_dict   

#------------------------------------------------------
subset = 'recipes'
subset_dict = {}
subset_dict['cuisine']=[]
subset_dict['recipe']=['rice_pudding', 'quiche']
subset_dict['preparation_time']=[]
subset_dict['cooking_time']=[]
subset_dict['ingredient']=['spinach', 'milk', '']
subset_dict['posting_date']=['2004', '2003', '2010']
subset_dict['meal']=['lunch', 'dinner']
all_dicts[subset] = subset_dict

#-----------------------------------------------------
subset = 'restaurants'
subset_dict = {}
subset_dict['neighborhood']=['midtown_west', 'chelsea']
subset_dict['reviews']=['30', '40']
subset_dict['reservations']=['true']
subset_dict['cuisine']=['thai', 'italian']
subset_dict['restaurant']=['thai_cafe', 'pizzeria_juno']
subset_dict['credit_cards']=['true']
subset_dict['delivery']=['true']
subset_dict['meal']=['lunch', 'dinner']
subset_dict['kids']=['true']
subset_dict['star_rating']=['2', '3', '4', '5']
subset_dict['groups']=['true']
subset_dict['price_rating']=['2', '3', '4', '5']
subset_dict['takeout']=['true']
subset_dict['outdoor_seating']=['true']
subset_dict['waiter_service']=['true']
all_dicts[subset] = subset_dict

np.save('all_dicts.npy',all_dicts)
#------------------------------------------------------

