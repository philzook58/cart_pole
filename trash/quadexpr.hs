
data Vec = X Int | U Int | 
data QuadTerm a b c = Quad Vec HMat Vec | Lin HVec Vec | Const Double 
type QuadExpr = [QuadTerm]

data LinTerm = MatVec HMat Vec | VecConst HVec
type LinExpr = [LinTerm]
-- it might be nice to be able to order the quadterms by time as part of the type. But there will be terms going across time?
-- maybe not


type Cost = QuadExpr

min :: Vec -> Cost -> (LinExpr, QuadTerm) 
min x expr= (solx ,   subs x solx expr)   where solx = solve x (diffterm x expr) 


diffterm :: Vec -> QuadTerm -> LinExpr
diffterm x (Quad y a z) = (if x == y then [MatVec a z] else []) ++ (if x == z then [MatVec a y] else [])
diffterm x (Lin h y) = if x == y then [VecConst h] else []
diffterm x (Const _) = []

simplifyQ :: QuadExpr -> QuadExpr
simplifyL :: LinExpr -> LinExpr



solve :: Vec -> LinExpr -> LinExpr



subs :: Vec -> LinExpr -> QuadExpr
subs 


-- this is all straightforward
-- but the Haskell principle
-- is that I should subvert the machinery of the language
-- to nefarious ends

-- make X and U types
--of Kind Vec
-- use typeclasses to solve?