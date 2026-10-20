"""Bounded exact grading of structured answers. Never executes candidate code."""
import ast
import json
from pathlib import Path
import sympy as s

ALLOWED_NAMES = frozenset({'ws','wf','n','r','v','s','x','y','h','alpha','delta','sigma2','t','q','K','epsilon','p','b','omega'})
MAX_TEXT = 1500
MAX_NODES = 180

class InvalidAnswer(ValueError):
    pass


def expression(value, assumptions=None):
    """Parse a small arithmetic grammar with exact constants and explicit symbols."""
    assumptions = assumptions or {}
    if not isinstance(assumptions,dict):
        raise InvalidAnswer('Invalid trusted symbol assumptions')
    for name, conditions in assumptions.items():
        if (name not in ALLOWED_NAMES or not isinstance(conditions,dict)
                or len(conditions)!=1 or not set(conditions)<={'positive','nonnegative'}
                or any(type(v) is not bool or not v for v in conditions.values())):
            raise InvalidAnswer('Unsupported trusted symbol assumption')
    if isinstance(value, bool) or not isinstance(value, (str, int, float)):
        raise InvalidAnswer('Expected a number or mathematical expression string')
    source = str(value).strip().replace('^', '**')
    if not source or len(source) > MAX_TEXT:
        raise InvalidAnswer('Expression length limit')
    try:
        tree = ast.parse(source, mode='eval')
    except (SyntaxError, RecursionError) as error:
        raise InvalidAnswer('Invalid expression') from error
    if sum(1 for _ in ast.walk(tree)) > MAX_NODES:
        raise InvalidAnswer('Expression complexity limit')
    denominators = []
    def visit(node):
        if isinstance(node, ast.Constant) and type(node.value) in (int, float):
            if len(str(node.value)) > 64:
                raise InvalidAnswer('Constant size limit')
            result = s.Rational(str(node.value))
        elif isinstance(node, ast.Name) and node.id in ALLOWED_NAMES:
            result = s.Symbol(node.id, **assumptions.get(node.id,{}))
        elif isinstance(node, ast.UnaryOp) and isinstance(node.op, (ast.UAdd, ast.USub)):
            result = visit(node.operand)
            if isinstance(node.op, ast.USub):
                result = -result
        elif isinstance(node, ast.BinOp):
            left,right = visit(node.left),visit(node.right)
            if isinstance(node.op, ast.Add):
                result = left+right
            elif isinstance(node.op, ast.Sub):
                result = left-right
            elif isinstance(node.op, ast.Mult):
                result = left*right
            elif isinstance(node.op, ast.Div):
                if right == 0:
                    raise InvalidAnswer('Division by zero')
                denominators.append(right)
                result = left/right
            elif isinstance(node.op, ast.Pow):
                integer_power = right.is_Integer and abs(right)<=16
                constant_root_power = (right.is_Rational and right.q in (3,4) and abs(right.p)<=16
                               and left.is_Rational and left.is_positive)
                if not (integer_power or constant_root_power):
                    raise InvalidAnswer('Only small integer powers or bounded positive rational cube/fourth roots are allowed')
                if right < 0:
                    if left == 0:
                        raise InvalidAnswer('Division by zero')
                    denominators.append(left)
                if left.is_Rational and max(int(left.p).bit_length(),int(left.q).bit_length())*max(1,abs(int(right.p)))>8192:
                    raise InvalidAnswer('Power size limit')
                result = left**right
            else:
                raise InvalidAnswer('Unsupported arithmetic operation')
        elif (isinstance(node, ast.Call) and isinstance(node.func, ast.Name)
              and node.func.id=='sqrt' and len(node.args)==1 and not node.keywords):
            result = s.sqrt(visit(node.args[0]))
        elif (isinstance(node, ast.Call) and isinstance(node.func, ast.Name)
              and node.func.id=='cbrt' and len(node.args)==1 and not node.keywords):
            radicand = visit(node.args[0])
            if not (radicand.is_Rational and radicand.is_positive):
                raise InvalidAnswer('Cube roots require positive rational constants')
            result = radicand**s.Rational(1,3)
        else:
            raise InvalidAnswer('Unsupported expression syntax')
        if any(power.exp.is_Integer and abs(power.exp)>64 for power in result.atoms(s.Pow)):
            raise InvalidAnswer('Result exponent limit')
        if result.is_Rational and max(int(result.p).bit_length(),int(result.q).bit_length())>8192:
            raise InvalidAnswer('Result integer size limit')
        if result.has(s.zoo,s.oo,-s.oo,s.nan):
            raise InvalidAnswer('Expression is not finite')
        return result
    return visit(tree.body),denominators


def denominator_factors(denominators):
    factors=set()
    for denominator in denominators:
        # A trusted domain can prove a radical denominator nonzero. Do not
        # infer domains from candidate text or cancel an unproved singularity.
        if denominator.is_nonzero is True:
            continue
        numerator=s.fraction(s.cancel(denominator))[0]
        if numerator.free_symbols:
            try:
                for factor,_ in s.factor_list(numerator)[1]:
                    symbols=sorted(factor.free_symbols,key=str)
                    polynomial=s.Poly(factor,*symbols).monic().as_expr()
                    factors.add(s.srepr(polynomial))
            except s.PolynomialError as error:
                raise InvalidAnswer('Unsupported symbolic denominator') from error
    return factors


def equivalent(candidate,expected,assumptions=None):
    if isinstance(expected,bool):
        return type(candidate) is bool and candidate is expected
    if isinstance(expected,dict):
        return (isinstance(candidate,dict) and set(candidate)==set(expected)
                and all(equivalent(candidate[k],v,assumptions) for k,v in expected.items()))
    if isinstance(expected,list):
        return (isinstance(candidate,list) and len(candidate)==len(expected)
                and all(equivalent(a,b,assumptions) for a,b in zip(candidate,expected)))
    a,ad=expression(candidate,assumptions);b,bd=expression(expected,assumptions)
    if a.free_symbols-b.free_symbols:
        return False
    # Reject removable new singularities instead of silently cancelling them.
    if not denominator_factors(ad)<=denominator_factors(bd):
        return False
    return s.simplify(a-b)==0


def grade(candidate,expected,assumptions=None):
    try:
        return {'correct':bool(equivalent(candidate,expected,assumptions)), 'valid':True}
    except (InvalidAnswer,ValueError,TypeError,ZeroDivisionError,RecursionError) as error:
        return {'correct':False,'valid':False,'error':str(error)}


def load_spec(row_id):
    path=Path(__file__).parent/'metadata.jsonl'
    for line in path.read_text().splitlines():
        row=json.loads(line)
        if row['id']==row_id:
            return row['canonical_expected'],row.get('symbol_assumptions',{})
    raise KeyError(row_id)


def load_expected(row_id):
    return load_spec(row_id)[0]


if __name__=='__main__':
    import sys
    if len(sys.argv)!=3:
        raise SystemExit('Usage: python grade_answers.py ROW_ID candidate.json')
    raw=Path(sys.argv[2]).read_text()
    if len(raw)>30000:
        raise SystemExit('Candidate answer size limit')
    candidate=json.loads(raw)
    expected,assumptions=load_spec(sys.argv[1])
    print(json.dumps(grade(candidate,expected,assumptions),ensure_ascii=False))
