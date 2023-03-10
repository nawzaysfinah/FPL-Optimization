import pandas as pd
import sasoptpy as so
import requests
import os
import time
from subprocess import Popen, DEVNULL
from concurrent.futures import ProcessPoolExecutor
from functools import lru_cache


@lru_cache(maxsize=1)
def get_data():
    r = requests.get('https://fantasy.premierleague.com/api/bootstrap-static/')
    fpl_data = r.json()
    
    # Retrieve element (player) data
    element_data = pd.DataFrame(fpl_data['elements'])

    # Retrieve team data
    team_data = pd.DataFrame(fpl_data['teams'])
    
    elements_team = pd.merge(element_data, team_data, left_on='team', right_on='id')
    merged_data = elements_team
    merged_data.set_index(['id_x'], inplace=True)
    type_data = pd.DataFrame(fpl_data['element_types']).set_index(['id'])

    return {'merged_data': merged_data, 'team_data': team_data, 'type_data': type_data}


def solve_single_period_fpl(budget):
    data = get_data()
    merged_data = data['merged_data']
    team_data = data['team_data']
    type_data = data['type_data']
    # next_gw = data['next_gw'] # we do not have next_gw data because we are not using FPL Reviews data, only from FPL API

    players = merged_data.index.to_list()
    element_types = type_data.index.to_list()
    teams = team_data['name'].to_list()

    model =so.Model(name='single_period')

    # Variables
    squad = model.add_variables(players, name='squad',vartype=so.binary)
    lineup = model.add_variables(players, name='lineup',vartype=so.binary)
    captain = model.add_variables(players, name='captain',vartype=so.binary)
    vicecap = model.add_variables(players, name='vicecap',vartype=so.binary)

    # Constraints
    squad_count = so.expr_sum(squad[p] for p in players)
    model.add_constraint(squad_count == 15, name ="squad_count")
    model.add_constraint(so.expr_sum(lineup[p] for p in players) == 11, name='lineup_count')
    model.add_constraint(so.expr_sum(captain[p] for p in players) == 1, name='captain_count')
    model.add_constraint(so.expr_sum(vicecap[p] for p in players) == 1, name='vicecap_count')
                         
    lineup_type_count = {t: so.expr_sum(lineup[p] for p in players if merged_data.loc[p,'element_type'] == t) for t in element_types}
    squad_type_count = {t: so.expr_sum(squad[p] for p in players if merged_data.loc[p,'element_type'] == t) for t in element_types}
    model.add_constraints((lineup_type_count[t] == [type_data.loc[t,'squad_min_play'],type_data.loc[t,'squad_max_play']] for t in element_types), name='valid_formation')
    model.add_constraints((squad_type_count[t] == type_data.loc[t,'squad_select'] for t in element_types), name='valid_squad')
    
    price = so.expr_sum(merged_data.loc[p, 'now_cost']/10 * squad[p] for p in players)
    model.add_constraint(price <= budget, name='budget_limit')
    model.add_constraints((so.expr_sum(squad[p] for p in players if merged_data.loc[p, 'name'] == t) <= 3 for t in teams), name='team_limit')
    merged_data['ep_next'] = merged_data['ep_next'].astype(float)
    
    # In Sertalps e.g., he uses FPL Review data that reads ExpPoints per Gameweek by allowing next_gw to be a variable to be read and calling the expected points as a function
    total_points = so.expr_sum(merged_data.loc[p, 'ep_next'] * (lineup[p]+ captain[p] + 0.1 * vicecap[p]) for p in players)
    
    # Remember for CBC, solves for minimising, so must -ve the total_points
    model.set_objective(-total_points, sense='N', name='total_xp')                   
    model.export_mps(f'single_period_{budget}.mps')

    command = f'cbc single_period_{budget}.mps solve solu solution_sp_{budget}.txt'
    # os.system(command)
    Popen(command,shell=False, stdout=DEVNULL).wait()

    for v in model.get_variables():
        v.set_value(0)

    with open('solution_sp_{budget}.txt', 'r') as f:
        for line in f:
            if 'objective value' in line:
                continue
            words = line.split()
            var = model.get_variable(words[1])
            var.set_value(float(words[2]))

    picks = []
    for p in players:
        if squad[p].get_value() > 0.5:
            lp = merged_data.loc[p]
            is_captain = 1 if captain[p].get_value() > 0.5 else 0
            is_lineup = 1 if lineup[p].get_value() > 0.5 else 0
            is_vice = 1 if vicecap[p].get_value() > 0.5 else 0
            picks.append([
                lp['web_name'], lp['element_type'], lp['name'], lp['now_cost']/10, round(lp['ep_next'], 2), is_lineup, is_captain, is_vice]
            )
            
    picks_df = pd.DataFrame(picks,columns=['name', 'pos','team', 'price', 'xP', 'lineup', 'captain', 'vicecaptain']).sort_values(by=['lineup', 'pos', 'xP'], ascending=[False,True,True])
    total_xp = so.expr_sum((lineup[p]+ captain[p]) * merged_data.loc[p,"ep_next"] for p in players).get_value()

    print(f'Total expected value for budget {budget}: {total_xp}')

    return {'model': model, 'picks': picks_df, 'total_xp': total_xp, 'price': price}


if __name__ == '__main__':
    
    # t0 = time.time()

    # results = []
    # for budget in range(80, 101, 5):
    #     r = solve_single_period_fpl(budget=budget)
    #     results.append([budget, r['total_xp']])

    # df = pd.DataFrame(results, columns=['budget','xP'])
    # print(df)

    # print(time.time()-t0, 'spent in for loop') 

    t0 = time.time()

    get_data()
    budget = list(range(80, 101, 5))

    with ProcessPoolExecutor(max_workers=16) as executor:
        responses = executor.map(solve_single_period_fpl, budget)
        all_xp_values = [r['total_xp'] for r in responses]
    results = zip(budget, all_xp_values)
    df = pd.DataFrame(results, columns=['budget','xP'])
    print(df)

    print(time.time()-t0, 'spent in for loop') 