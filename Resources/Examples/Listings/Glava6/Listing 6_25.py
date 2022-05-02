# Listing 6.25
X1 = np.array([[1.5, 1.5]])
X2 = np.array([[0.0, 0.0]])
X3 = np.array([[-1, -1]])
p1 = Lr.predict_proba(X1)
p2 = Lr.predict_proba(X2)
p3 = Lr.predict_proba(X3)

print(X1)
print(p1)
print(X2)
print(p2)
print(X3)
print(p3)
