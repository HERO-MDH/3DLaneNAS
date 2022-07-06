class db_entry:
    def __init__(self, train_loss, ll_f_measure, ll_recall, ll_precision,
                 cl_f_measure, cl_recal, cl_precision, latency):

        self.train_loss = train_loss
        self.ll_f_measure = ll_f_measure
        self.ll_recall = ll_recall
        self.ll_precision = ll_precision
        self.cl_f_measure = cl_f_measure
        self.cl_recal = cl_recal
        self.cl_precision = cl_precision
        self.latency = latency