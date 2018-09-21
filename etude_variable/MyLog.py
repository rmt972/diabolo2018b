import  sys
sys.path.insert(0, "C:/projets_python/diabolo")

import logging

import logging.config

logging.config.fileConfig('C:\projets_python\diabolo\etude_variable\diabolo.conf')


def traceLogInfo( mes, titre="diabolo"):
   print(mes)

   #logger = logging.getLogger(titre)
   #logger.info(mes)






def traceLogdebug(mes, titre ="diabolo"):


    logger = logging.getLogger(titre)
    logger.debug(mes)








