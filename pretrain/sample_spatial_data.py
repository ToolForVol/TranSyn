"""
@desc: This script aims to select genomic-close/control source samples to pretrain
@author: chen ye
@email: q23101020@stu.edu.cn
"""


from datetime import datetime
import pyranges as pr
import pandas as pd
import os
import numpy as np


def df_to_ranges(df):
    """Convert input dataframe to PyRanges object"""
    df = df.rename(columns={'#CHROM': 'Chromosome', 'POS': 'Start'})
    df['End'] = df['Start']  # Single nucleotide variation
    return pr.PyRanges(df[['Chromosome', 'Start', 'End']].copy())


def retrieve_link(df, chrom_col, pos_col, prefix=''):
    """Create a unique identifier link for mutations"""
    link = [f'{prefix}{chro}_{pos}' for chro, pos in zip(df[chrom_col], df[pos_col])]
    return link


def retrieve_close_samples(source_neg, source_pos, target, distance):
    """
    Desc:
        Find samples near the target domain within a specific genomic distance.
    Args:
        source_neg: Source domain negative samples DataFrame.
        source_pos: Source domain positive samples DataFrame.
        target: Target domain DataFrame.
        distance: Distance threshold to select samples within 'distance' bp of target context.
    Returns:
        source_pos_near: Selected mutations + start/end positions within distance (Positive).
        source_neg_near: Selected mutations + start/end positions within distance (Negative).
        source_pos_choose: Selected mutation metadata (Positive).
        source_neg_choose: Selected mutation metadata (Negative).
    """
    # Convert to PyRanges
    target_ranges = df_to_ranges(target)
    source_pos_ranges = df_to_ranges(source_pos)
    source_neg_ranges = df_to_ranges(source_neg)

    # Set +/- distance range by expanding target intervals
    target_buffered = target_ranges.slack(distance)  # slack() automatically expands Start/End

    # Find overlapping regions
    pos_near = source_pos_ranges.join(target_buffered)
    neg_near = source_neg_ranges.join(target_buffered)

    # Convert back to DataFrame
    source_pos_near = pos_near.df
    source_neg_near = neg_near.df

    # Match back to original source data using link IDs
    source_pos_near['link'] = retrieve_link(source_pos_near, 'Chromosome', 'Start')
    source_neg_near['link'] = retrieve_link(source_neg_near, 'Chromosome', 'Start')
    source_pos['link'] = retrieve_link(source_pos, '#CHROM', 'POS')
    source_neg['link'] = retrieve_link(source_neg, '#CHROM', 'POS')
    
    source_pos_choose = source_pos[source_pos['link'].isin(source_pos_near['link'])]
    source_neg_choose = source_neg[source_neg['link'].isin(source_neg_near['link'])]

    return source_pos_near, source_neg_near, source_pos_choose, source_neg_choose


def retrieve_far_samples(
    source_neg,
    source_pos,
    target,
    save_dir,
    min_distance=2000,
    n=1000,
    random_state=913
):
    """
    Desc:
        Select mutations located at least 'min_distance' away from the target,
        and randomly sample 'n' instances (split between positive and negative).
    Args:
        source_neg: Source domain negative samples DataFrame.
        source_pos: Source domain positive samples DataFrame.
        target: Target domain DataFrame.
        min_distance: Minimum distance (default 2000bp).
        n: Total number of samples to randomly draw.
        random_state: Random seed.
    Returns:
        None (Saves source_pos_far_sampled and source_neg_far_sampled to CSV).
    """
    # Convert to PyRanges
    target_ranges = df_to_ranges(target)
    source_pos_ranges = df_to_ranges(source_pos)
    source_neg_ranges = df_to_ranges(source_neg)
    
    # Construct target ± min_distance buffer
    target_buffered = target_ranges.slack(min_distance)
    
    # Identify "nearby" mutations to exclude
    pos_near = source_pos_ranges.join(target_buffered)
    neg_near = source_neg_ranges.join(target_buffered)
    pos_near_df = pos_near.df
    neg_near_df = neg_near.df
    
    # Construct links for filtering
    pos_near_df["link"] = retrieve_link(pos_near_df, "Chromosome", "Start")
    neg_near_df["link"] = retrieve_link(neg_near_df, "Chromosome", "Start")
    source_pos["link"] = retrieve_link(source_pos, "#CHROM", "POS")
    source_neg["link"] = retrieve_link(source_neg, "#CHROM", "POS")
    
    # Inverse selection: Remove mutations within the min_distance buffer
    source_pos_far = source_pos[~source_pos["link"].isin(pos_near_df["link"])]
    source_neg_far = source_neg[~source_neg["link"].isin(neg_near_df["link"])]
    
    # Random sampling
    source_pos_far_sampled = source_pos_far.sample(n=n//2, random_state=random_state)
    source_neg_far_sampled = source_neg_far.sample(n=n//2, random_state=random_state)
    
    source_pos_far_sampled.to_csv(f"{save_dir}/pos.txt", sep='\t', index=False)
    source_neg_far_sampled.to_csv(f"{save_dir}/neg.txt", sep='\t', index=False)


def retrieve_based_on_granularity(source_pos_path, source_neg_path, target_path, save_dir):
    """Extract information based on different distance granularities"""
    print(f"[INFO] {datetime.now().strftime('%Y-%m-%d %H:%M:%S')} | Start screening...")
    
    # Load Source Domain
    source_pos = pd.read_csv(source_pos_path, sep='\t')
    source_neg = pd.read_csv(source_neg_path, sep='\t')
    
    # Remove mutual exclusion within source domain
    pos_vids = set(source_pos['variant38'])
    neg_vids = set(source_neg['variant38'])
    source_pos = source_pos[~source_pos['variant38'].isin(neg_vids)]
    source_neg = source_neg[~source_neg['variant38'].isin(pos_vids)]
    
    # Load Target Domain
    target = pd.read_csv(target_path, sep='\t')
    target_vids = set(target['variant_hg38'])
    
    # Remove domain-specific exclusion (Source vs Target)
    source_pos = source_pos[~source_pos['variant38'].isin(target_vids)]
    source_neg = source_neg[~source_neg['variant38'].isin(target_vids)]
    
    # Define distance range
    distance_list = list(range(500, 5500, 500))
    pos_count_list = []
    neg_count_list = []
    
    for distance in distance_list:
        # Retrieve nearby mutation information
        _, _, source_pos_choose, source_neg_choose = retrieve_close_samples(source_neg, source_pos, target, distance)
        
        print(f"[INFO] Selecting area within {distance}bp")
        tmp_save_dir = f"{save_dir}/near_{distance}bp/"
        os.makedirs(tmp_save_dir, exist_ok=True)
        
        source_pos_choose.to_csv(f"{tmp_save_dir}/pos.txt", sep='\t', index=False)
        source_neg_choose.to_csv(f"{tmp_save_dir}/neg.txt", sep='\t', index=False)
        
        pos_count_list.append(len(source_pos_choose))
        neg_count_list.append(len(source_neg_choose))
        
        print(f"[INFO] Positive samples count: {len(source_pos_choose)}")
        print(f"[INFO] Negative samples count: {len(source_neg_choose)}")
        
    # Compile statistics
    stat_df = pd.DataFrame({
        'Distance': distance_list, 
        'Source_sample_amount_positive': pos_count_list,
        'Source_sample_amount_negative': neg_count_list
    })
    stat_df['Source_sample_amount'] = stat_df['Source_sample_amount_positive'] + stat_df['Source_sample_amount_negative']
    stat_df.to_csv(f"{save_dir}/source_amount.info", sep='\t', index=False)
    
    print(f"[INFO] {datetime.now().strftime('%Y-%m-%d %H:%M:%S')} | Done!")