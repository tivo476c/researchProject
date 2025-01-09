* why t=20 in Method3_Sethian / function all_my_midpoints
* is this a problem: clean_and_collect_my_vertices iterates just over N-1 cells (i think its not a 
  problem since all vertices should already be cleaned whith the second last cell)
  when iterating over N cells, there is an error in the last loop because the dimension of dirty_i 
  is wrong when doing k>i in method3_sethian_saye.py 