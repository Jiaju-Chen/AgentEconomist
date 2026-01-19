from ..model import Transaction
from agentsociety_ecosim.utils.log_utils import setup_global_logger
logger = setup_global_logger(__name__)

__all__ = ["consume_tax_middleware", "labor_tax_middleware", "VAT_tax_middleware"]

def consume_tax_middleware(rate: float, gov_id: str):
    def apply_consume_tax(tx: Transaction, ledger):
        tax = tx.amount * rate
        tx.amount -= tax
        ledger[gov_id].amount += tax
        logger.info(f"[tax] {rate*100:.1f}%: {gov_id} receive {tax:.2f}")
    return apply_consume_tax

# calculate labor tax and income to labor
def labor_tax_middleware(rate: float, gov_id: str):
    def apply_labor_tax(tx: Transaction, ledger):
        tax = tx.amount * rate # labor tax
        tx.amount -= tax # income to labor
        ledger[gov_id].amount += tax
        logger.info(f"[tax] {rate*100:.1f}%: {gov_id} receive {tax:.2f}")
    return apply_labor_tax


def VAT_tax_middleware(rate: float, gov_id: str):
    def apply_VAT_tax(tx: Transaction, ledger):
        tax = tx.amount * rate
        tx.amount -= tax
        ledger[gov_id].amount += tax
        logger.info(f"[tax] {rate*100:.1f}%: {gov_id} receive {tax:.2f}")
    return apply_VAT_tax

# Example usage
# center = EconomicCenter.remote()
# ray.get(center.register_middleware.remote('purchase',consume_tax_middleware(0.1, "gov")))
# ray.get(center.register_middleware.remote('labor', labor_tax_middleware(0.1, "gov")))
