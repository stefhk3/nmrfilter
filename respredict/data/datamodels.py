

"""
Simplified schema for molecules, atoms, spectra
"""

from sqlalchemy import *

import sys; sys.path.append("../")

import dbconfig

def create():

    metadata = MetaData()

    molecule = Table('molecule', metadata,
                     Column('id', Integer, primary_key=True),
                     Column('name', Text), # convenience name, 
                     Column('cas_id', Text, index=True), # CAS ID if known
                     Column("pubchem_cid", Integer, index=True), # pubchem cas ID
                     Column("smiles", Text, index=True), # smiles string, 
                     Column("inchi", Text, index=True), # smiles string, 
                     Column("source", Text, index=True, nullable=False),  # ID from wherever we got this molecule 
                     Column("source_id", Text, index=True, nullable=False),  # ID from wherever we got this molecule 
                     Column("mol", Text, nullable=False), # "MOL description of molecule topology", 
                     Column("meta", JSON),
                     UniqueConstraint('source', 'source_id', name='source')
    )

    Index('molecule_meta_gin', molecule.c.meta, postgresql_using='gin')

    atom = Table("atom", metadata, 
                  Column("id", Integer, primary_key=True), 
                  Column("molecule_id", Integer, ForeignKey("molecule.id"), index=True), 
                  Column("idx", Integer, nullable=False), # pos in mol file
                  Column("atomicnum", Integer, nullable=False, index=True), # atomic number, 
                  )

    """
    We assume here that spectra are assignments of peaks to molecules
    """

    spectrum_meta = Table("spectrum_meta", metadata, 
                          Column("id", Integer, primary_key=True), 
                          Column("molecule_id", Integer, ForeignKey("molecule.id"), nullable=False, index=True), 
                          Column("solvent", Text, index=True),
                          Column("nucleus", Text, index=True),
                          Column("temp", Float), 
                          Column("field", Float), 
                          Column("source", Text, index=True, nullable=False),  # ID from wherever we got this spectrum
                          Column("source_id", Text, index=True, nullable=False),  # ID from wherever we got this spectrum 
                          Column("meta", JSON),
                          UniqueConstraint('source', 'source_id', name='spec_source')
    )

    Index('spectrum_meta_gin', spectrum_meta.c.meta, postgresql_using='gin')

    peak = Table("peak", metadata, 
                 Column("id", Integer, primary_key=True), 
                 Column("spectrum_id", Integer, ForeignKey("spectrum_meta.id"), index=True), 
                 Column("atom_id", Integer, ForeignKey("atom.id")), 
                 Column("multiplicity", Text), 
                 Column("shape", Text), 
                 Column("units", Text, nullable=False), 
                 Column("value", Float, nullable=False))


    engine = create_engine(dbconfig.AWS_RDS_DB_STR)
    metadata.create_all(engine) 

if __name__ == "__main__":
    create()
