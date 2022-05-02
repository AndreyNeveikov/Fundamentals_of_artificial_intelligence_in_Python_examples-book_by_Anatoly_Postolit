# Listing 6.5
for mod in net.modules:
  print("Module:", mod.name)
  if mod.paramdim > 0:
    print("--parameters:", mod.params)
  for conn in net.connections[mod]:
    print("-connection to", conn.outmod.name)
    if conn.paramdim > 0:
      print("- parameters", conn.params)
  if hasattr(net, "recurrentConns"):
    print("Recurrent connections")
    for conn in net.recurrentConns:
      print("-", conn.inmod.name, " to", conn.outmod.name)
      if conn.paramdim > 0:
        print("- parameters", conn.params)
